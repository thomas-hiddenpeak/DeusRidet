// text-output-panel.js — Conversation text output panel
//
// Shows 黑娃's decode output as text. Includes a text input for
// direct communication. Displays both user input and entity responses
// in a chat-like timeline.

export class TextOutputPanel {
    constructor(ws) {
        this.ws = ws;
        this.el = document.getElementById('text-output-panel');
        if (!this.el) return;

        this.messages = [];   // [{role, text, time, state, tokens, ms}]
        this.maxMessages = 200;

        this._render();
        this._bindEvents();
    }

    _render() {
        this.el.innerHTML = `
            <h2 class="panel__title">Dialogue</h2>
            <div class="to-messages" id="to-messages"></div>
            <div class="to-input-wrap">
                <input type="text" id="to-input" class="to-input"
                       placeholder="Send text to 黑娃..." autocomplete="off">
                <button id="to-send" class="btn btn--mic">Send</button>
            </div>
        `;

        this.messagesEl = this.el.querySelector('#to-messages');
        this.inputEl = this.el.querySelector('#to-input');
        this.sendBtn = this.el.querySelector('#to-send');
    }

    _bindEvents() {
        if (!this.inputEl || !this.sendBtn) return;

        this.sendBtn.addEventListener('click', () => this._send());
        this.inputEl.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this._send();
            }
        });
    }

    _send() {
        const text = this.inputEl.value.trim();
        if (!text) return;

        // Send to server
        this.ws.sendText('text_input:' + text);

        // Add to local chat
        this._addMessage({
            role: 'user',
            text: text,
            time: new Date().toLocaleTimeString('en-GB', { hour12: false })
        });

        this.inputEl.value = '';
        this.inputEl.focus();
    }

    // Called on consciousness_decode messages (final complete speech)
    onDecode(data) {
        if (!this.el) return;
        // If we have a streaming element, finalize it with full text + metadata
        if (this._streamEl) {
            this._streamEl.classList.remove('to-msg--streaming');
            const textEl = this._streamEl.querySelector('.to-msg-text');
            if (textEl) textEl.textContent = data.text;
            const metaEl = this._streamEl.querySelector('.to-msg-meta');
            if (metaEl && data.tokens) {
                const time = new Date().toLocaleTimeString('en-GB', { hour12: false });
                const stateClass = data.state ? ' cs-state--' + data.state : '';
                metaEl.innerHTML =
                    `<span class="to-time">${time}</span>` +
                    `<span class="to-state${stateClass}">${(data.state || '').toUpperCase()}</span>` +
                    `<span class="to-meta">${data.tokens}tok ${data.time_ms?.toFixed(0)}ms</span>`;
            }
            this._streamEl = null;
            return;
        }
        // Fallback: no streaming happened, add as complete message
        this._addMessage({
            role: 'entity',
            text: data.text,
            state: data.state,
            tokens: data.tokens,
            ms: data.time_ms,
            time: new Date().toLocaleTimeString('en-GB', { hour12: false })
        });
    }

    // Called per-token during speech decode for streaming display
    onSpeechToken(data) {
        if (!this.el || !this.messagesEl) return;
        // token_id == -1 signals start of new speech
        if (data.token_id === -1) {
            const div = document.createElement('div');
            div.className = 'to-msg to-msg--entity to-msg--streaming';
            const time = new Date().toLocaleTimeString('en-GB', { hour12: false });
            div.innerHTML = `
                <div class="to-msg-meta"><span class="to-time">${time}</span><span class="to-state cs-state--active">SPEAKING...</span></div>
                <div class="to-msg-text"></div>
            `;
            this.messagesEl.appendChild(div);
            this._streamEl = div;
            this._streamText = '';
            this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
            return;
        }
        // Append token text
        if (this._streamEl) {
            this._streamText += data.text;
            const textEl = this._streamEl.querySelector('.to-msg-text');
            if (textEl) textEl.textContent = this._streamText;
            this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
        }
    }

    // Called on ASR transcript (show what was heard)
    onAsrTranscript(data) {
        if (!this.el) return;
        const speaker = data.speaker_name || ('Speaker #' + (data.speaker_id >= 0 ? data.speaker_id : '?'));
        this._addMessage({
            role: 'asr',
            text: data.text,
            speaker: speaker,
            time: new Date().toLocaleTimeString('en-GB', { hour12: false })
        });
    }

    _addMessage(msg) {
        this.messages.push(msg);
        if (this.messages.length > this.maxMessages) {
            this.messages.shift();
        }
        this._renderLast(msg);
    }

    _renderLast(msg) {
        if (!this.messagesEl) return;

        const div = document.createElement('div');
        div.className = 'to-msg to-msg--' + msg.role;

        let meta = `<span class="to-time">${msg.time}</span>`;
        if (msg.role === 'entity') {
            const stateClass = msg.state ? ' cs-state--' + msg.state : '';
            meta += `<span class="to-state${stateClass}">${(msg.state || '').toUpperCase()}</span>`;
            if (msg.tokens) meta += `<span class="to-meta">${msg.tokens}tok ${msg.ms?.toFixed(0)}ms</span>`;
        }
        if (msg.role === 'asr') {
            meta += `<span class="to-speaker">${this._esc(msg.speaker)}</span>`;
        }

        div.innerHTML = `
            <div class="to-msg-meta">${meta}</div>
            <div class="to-msg-text">${this._esc(msg.text)}</div>
        `;

        this.messagesEl.appendChild(div);
        this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    }

    _esc(s) {
        if (!s) return '';
        const el = document.createElement('span');
        el.textContent = s;
        return el.innerHTML;
    }
}
