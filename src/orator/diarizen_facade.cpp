/**
 * @file diarizen_facade.cpp
 * @philosophical_role Glue between the C++ Orator and the Python DiariZen-v2
 *     worker. POSIX pipes + fork/execvp + a tiny hand-rolled JSON reader
 *     limited to the worker's known reply shape. Not a generic IPC layer.
 * @serves DiarizenFacade. No callers outside src/orator/.
 */
#include "orator/diarizen_facade.h"

#include "communis/json_util.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace deusridet::orator {

namespace {

// Minimal whitespace-skipping JSON cursor — only what we need to read
// `{"ok": bool, "segments": [[num,num,"str"], ...], ...}`. Aborts on
// surprises by setting an error string; the caller treats any parse
// failure as a worker malfunction.
struct JCursor {
    const char* p = nullptr;
    const char* end = nullptr;
    std::string err;

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' ||
                           *p == '\r' || *p == ','))
            ++p;
    }
    bool eat(char c) {
        skip_ws();
        if (p < end && *p == c) { ++p; return true; }
        return false;
    }
    bool match_keyword(const char* kw) {
        skip_ws();
        size_t n = std::strlen(kw);
        if (p + n > end) return false;
        if (std::memcmp(p, kw, n) != 0) return false;
        p += n;
        return true;
    }
    bool read_string(std::string& out) {
        skip_ws();
        if (p >= end || *p != '"') { err = "expected string"; return false; }
        ++p;
        out.clear();
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) {
                char c = p[1];
                switch (c) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case '/': out += '/'; break;
                    default:
                        // Drop unknown escapes (incl. \uXXXX — labels are ASCII).
                        out += c;
                        break;
                }
                p += 2;
            } else {
                out += *p++;
            }
        }
        if (p >= end) { err = "unterminated string"; return false; }
        ++p;
        return true;
    }
    bool read_number(double& out) {
        skip_ws();
        char* nend = nullptr;
        out = std::strtod(p, &nend);
        if (nend == p) { err = "expected number"; return false; }
        p = nend;
        return true;
    }
    // Skip a JSON value (number / string / bool / null / array / object).
    // Used to step past keys we don't care about.
    bool skip_value() {
        skip_ws();
        if (p >= end) return false;
        char c = *p;
        if (c == '"') { std::string tmp; return read_string(tmp); }
        if (c == '[' || c == '{') {
            char open = c, close = (c == '[') ? ']' : '}';
            ++p;
            int depth = 1;
            bool in_str = false;
            while (p < end && depth > 0) {
                char x = *p++;
                if (in_str) {
                    if (x == '\\' && p < end) { ++p; }
                    else if (x == '"') in_str = false;
                } else if (x == '"') in_str = true;
                else if (x == open) ++depth;
                else if (x == close) --depth;
            }
            return depth == 0;
        }
        // bareword: number / true / false / null
        while (p < end && *p != ',' && *p != '}' && *p != ']' &&
               *p != ' ' && *p != '\n' && *p != '\r' && *p != '\t')
            ++p;
        return true;
    }
};

// Read one '\n'-terminated line from fd with a deadline (ms). Returns the
// line WITHOUT the trailing '\n'. On timeout/EOF/error returns std::nullopt
// equivalent: false + sets `err`.
bool read_line_with_timeout(int fd, std::string& line, int timeout_ms,
                            std::string& err) {
    line.clear();
    const int chunk = 4096;
    std::string buf;
    buf.reserve(chunk);
    long deadline_ms = timeout_ms;
    while (true) {
        // Try non-blocking read first.
        char tmp[chunk];
        ssize_t n = ::read(fd, tmp, sizeof(tmp));
        if (n > 0) {
            for (ssize_t i = 0; i < n; ++i) {
                if (tmp[i] == '\n') {
                    if (i + 1 < n) {
                        // Stash leftover; the worker only sends one line per
                        // reply, but be defensive against double-line bursts
                        // from earlier protocol noise.
                        err = "unexpected extra bytes after newline";
                    }
                    return true;
                }
                line += tmp[i];
            }
            continue;
        }
        if (n == 0) { err = "worker EOF"; return false; }
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            err = std::string("read: ") + std::strerror(errno);
            return false;
        }
        struct pollfd pfd { fd, POLLIN, 0 };
        int poll_chunk = (deadline_ms < 0) ? -1
                         : (deadline_ms > 1000 ? 1000 : (int)deadline_ms);
        int pr = ::poll(&pfd, 1, poll_chunk);
        if (pr < 0) {
            if (errno == EINTR) continue;
            err = std::string("poll: ") + std::strerror(errno);
            return false;
        }
        if (pr == 0) {
            if (deadline_ms < 0) continue;
            deadline_ms -= poll_chunk;
            if (deadline_ms <= 0) { err = "timeout"; return false; }
            continue;
        }
        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
            // Still try one more read — there may be buffered bytes.
            continue;
        }
    }
}

std::string default_worker_script() {
    namespace fs = std::filesystem;
    fs::path candidates[] = {
        fs::current_path() / "tools" / "diarizen_worker.py",
        fs::path("/home/rm01/DeusRidet/tools/diarizen_worker.py"),
    };
    for (const auto& p : candidates) {
        std::error_code ec;
        if (fs::exists(p, ec)) return p.string();
    }
    return candidates[0].string();
}

}  // namespace

struct DiarizenFacade::Impl {
    DiarizenFacadeConfig cfg;
    pid_t pid = 0;
    int  in_fd  = -1;   // parent writes here -> worker stdin
    int  out_fd = -1;   // parent reads here  <- worker stdout
    bool started = false;
    bool loaded  = false;
    std::string last_err;

    ~Impl() { hard_kill(); }

    void hard_kill() noexcept {
        if (in_fd  >= 0) { ::close(in_fd);  in_fd  = -1; }
        if (out_fd >= 0) { ::close(out_fd); out_fd = -1; }
        if (pid > 0) {
            int status = 0;
            // Give it a moment, then SIGKILL if still alive.
            for (int i = 0; i < 5; ++i) {
                pid_t r = ::waitpid(pid, &status, WNOHANG);
                if (r == pid) { pid = 0; return; }
                if (r < 0)    { pid = 0; return; }
                ::usleep(100'000);
            }
            ::kill(pid, SIGKILL);
            ::waitpid(pid, &status, 0);
            pid = 0;
        }
    }

    bool spawn() {
        if (started) return true;
        std::string script = cfg.worker_script.empty()
                                 ? default_worker_script()
                                 : cfg.worker_script;
        int pipe_in[2]  = { -1, -1 };  // parent -> child stdin
        int pipe_out[2] = { -1, -1 };  // child  -> parent stdout
        if (::pipe(pipe_in) != 0 || ::pipe(pipe_out) != 0) {
            last_err = std::string("pipe: ") + std::strerror(errno);
            if (pipe_in[0]  >= 0) ::close(pipe_in[0]);
            if (pipe_in[1]  >= 0) ::close(pipe_in[1]);
            if (pipe_out[0] >= 0) ::close(pipe_out[0]);
            if (pipe_out[1] >= 0) ::close(pipe_out[1]);
            return false;
        }
        pid_t child = ::fork();
        if (child < 0) {
            last_err = std::string("fork: ") + std::strerror(errno);
            ::close(pipe_in[0]);  ::close(pipe_in[1]);
            ::close(pipe_out[0]); ::close(pipe_out[1]);
            return false;
        }
        if (child == 0) {
            // Child: wire stdin <- pipe_in[0], stdout -> pipe_out[1].
            // Leave stderr inherited so library noise is visible in logs.
            ::dup2(pipe_in[0], 0);
            ::dup2(pipe_out[1], 1);
            ::close(pipe_in[0]);  ::close(pipe_in[1]);
            ::close(pipe_out[0]); ::close(pipe_out[1]);
            // Unbuffered Python so each reply lands on the wire immediately.
            const char* argv[] = {
                cfg.python_bin.c_str(),
                "-u",
                script.c_str(),
                nullptr,
            };
            ::execvp(argv[0], const_cast<char* const*>(argv));
            // execvp returned -> failure. Surface and die so parent EOFs.
            std::fprintf(stderr, "[diarizen_facade] execvp(%s) failed: %s\n",
                         argv[0], std::strerror(errno));
            std::_Exit(127);
        }
        // Parent.
        ::close(pipe_in[0]);
        ::close(pipe_out[1]);
        in_fd = pipe_in[1];
        out_fd = pipe_out[0];
        // Make read end non-blocking; reader uses poll() for timeout.
        int fl = ::fcntl(out_fd, F_GETFL, 0);
        ::fcntl(out_fd, F_SETFL, fl | O_NONBLOCK);
        pid = child;
        started = true;
        return true;
    }

    bool send_line(const std::string& line) {
        std::string buf = line;
        if (buf.empty() || buf.back() != '\n') buf += '\n';
        const char* data = buf.data();
        size_t left = buf.size();
        while (left > 0) {
            ssize_t n = ::write(in_fd, data, left);
            if (n <= 0) {
                if (errno == EINTR) continue;
                last_err = std::string("write: ") + std::strerror(errno);
                return false;
            }
            data += n; left -= n;
        }
        return true;
    }

    bool recv_line(std::string& line, int timeout_sec) {
        std::string err;
        if (!read_line_with_timeout(out_fd, line, timeout_sec * 1000, err)) {
            last_err = "recv: " + err;
            return false;
        }
        return true;
    }

    bool greet() {
        // Worker writes {"ok":true,"ready":true,...} as its first line.
        std::string line;
        if (!recv_line(line, cfg.spawn_timeout_sec)) return false;
        if (line.find("\"ready\"") == std::string::npos) {
            last_err = "worker greeting unexpected: " + line;
            return false;
        }
        return true;
    }

    bool load_model() {
        if (loaded) return true;
        std::string req = "{\"op\":\"load\",\"model\":\"" +
                          communis::json_escape(cfg.model_name) + "\"}";
        if (!send_line(req)) return false;
        std::string line;
        // First-time model load: WavLM-large download + GPU upload. Keep a
        // generous bound (60 s on Orin is comfortable).
        if (!recv_line(line, 120)) return false;
        if (line.find("\"ok\": true") == std::string::npos &&
            line.find("\"ok\":true")  == std::string::npos) {
            last_err = "load failed: " + line;
            return false;
        }
        loaded = true;
        return true;
    }
};

DiarizenFacade::DiarizenFacade(DiarizenFacadeConfig cfg)
    : impl_(std::make_unique<Impl>()) {
    impl_->cfg = std::move(cfg);
}

DiarizenFacade::~DiarizenFacade() {
    if (impl_) shutdown();
}

DiarizenFacade::DiarizenFacade(DiarizenFacade&&) noexcept = default;
DiarizenFacade& DiarizenFacade::operator=(DiarizenFacade&&) noexcept = default;

bool DiarizenFacade::start() {
    if (!impl_) return false;
    if (impl_->started && impl_->loaded) return true;
    if (!impl_->started) {
        if (!impl_->spawn())   return false;
        if (!impl_->greet())   { impl_->hard_kill(); return false; }
    }
    if (!impl_->load_model())  return false;
    return true;
}

std::vector<DiarizenSegment>
DiarizenFacade::diarize(const std::string& wav_path) {
    std::vector<DiarizenSegment> out;
    if (!impl_) return out;
    if (!start()) return out;

    std::string req = "{\"op\":\"diarize\",\"wav\":\"" +
                      communis::json_escape(wav_path) + "\"}";
    if (!impl_->send_line(req)) return out;
    std::string line;
    if (!impl_->recv_line(line, impl_->cfg.diarize_timeout_sec)) return out;

    JCursor j;
    j.p = line.data(); j.end = line.data() + line.size();
    if (!j.eat('{')) { impl_->last_err = "diarize: no opening brace"; return out; }
    bool ok = false;
    bool has_ok = false;
    std::vector<DiarizenSegment> segs;
    bool has_segments = false;
    std::string error_msg;

    while (j.p < j.end) {
        j.skip_ws();
        if (j.p < j.end && *j.p == '}') { ++j.p; break; }
        std::string key;
        if (!j.read_string(key)) break;
        if (!j.eat(':')) break;
        if (key == "ok") {
            j.skip_ws();
            if (j.match_keyword("true"))       { ok = true;  has_ok = true; }
            else if (j.match_keyword("false")) { ok = false; has_ok = true; }
            else j.skip_value();
        } else if (key == "error") {
            j.read_string(error_msg);
        } else if (key == "segments") {
            j.skip_ws();
            if (!j.eat('[')) { j.skip_value(); continue; }
            has_segments = true;
            while (j.p < j.end) {
                j.skip_ws();
                if (j.p < j.end && *j.p == ']') { ++j.p; break; }
                if (!j.eat('[')) break;
                DiarizenSegment s;
                if (!j.read_number(s.start_sec)) break;
                if (!j.read_number(s.end_sec))   break;
                if (!j.read_string(s.label))     break;
                if (!j.eat(']')) break;
                segs.push_back(std::move(s));
            }
        } else {
            j.skip_value();
        }
    }
    if (!has_ok || !ok) {
        impl_->last_err = "diarize failed: " +
                          (error_msg.empty() ? line : error_msg);
        return out;
    }
    if (!has_segments) {
        impl_->last_err = "diarize: missing segments";
        return out;
    }
    out = std::move(segs);
    impl_->last_err.clear();
    return out;
}

void DiarizenFacade::shutdown() noexcept {
    if (!impl_ || !impl_->started) return;
    impl_->send_line("{\"op\":\"shutdown\"}");
    std::string line;
    impl_->recv_line(line, 5);  // best-effort
    impl_->hard_kill();
    impl_->started = false;
    impl_->loaded = false;
}

const std::string& DiarizenFacade::last_error() const noexcept {
    static const std::string empty;
    return impl_ ? impl_->last_err : empty;
}

int DiarizenFacade::worker_pid() const noexcept {
    return impl_ ? impl_->pid : 0;
}

}  // namespace deusridet::orator
