/**
 * @file log.h
 * @philosophical_role Structured logging primitive. Every log line is an observation the entity (or its operator) can later reread — logs are short-term memoranda, not debug noise.
 * @serves Every subsystem; no subsystem prints to stderr directly.
 */
// log.h — Structured logging with severity levels
//
// Simple, header-only logging for DeusRidet. Outputs to stderr with
// timestamp, severity, and module name.

#pragma once

#include <cstdio>
#include <ctime>
#include <cstdarg>

namespace deusridet {
namespace log {

enum class Level { DEBUG, INFO, WARN, ERROR };

inline const char* level_str(Level l) {
    switch (l) {
        case Level::DEBUG: return "DEBUG";
        case Level::INFO:  return "INFO";
        case Level::WARN:  return "WARN";
        case Level::ERROR: return "ERROR";
    }
    return "???";
}

inline Level g_min_level = Level::INFO;

inline void set_level(Level l) { g_min_level = l; }

inline void log(Level level, const char* module, const char* fmt, ...) {
    if (level < g_min_level) return;

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    localtime_r(&ts.tv_sec, &tm);

    fprintf(stderr, "[%02d:%02d:%02d.%03ld] [%5s] [%s] ",
            tm.tm_hour, tm.tm_min, tm.tm_sec, ts.tv_nsec / 1000000,
            level_str(level), module);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fputc('\n', stderr);
}

} // namespace log
} // namespace deusridet

#define LOG_DEBUG(mod, ...) ::deusridet::log::log(::deusridet::log::Level::DEBUG, mod, __VA_ARGS__)
#define LOG_INFO(mod, ...)  ::deusridet::log::log(::deusridet::log::Level::INFO,  mod, __VA_ARGS__)
#define LOG_WARN(mod, ...)  ::deusridet::log::log(::deusridet::log::Level::WARN,  mod, __VA_ARGS__)
#define LOG_ERROR(mod, ...) ::deusridet::log::log(::deusridet::log::Level::ERROR, mod, __VA_ARGS__)
