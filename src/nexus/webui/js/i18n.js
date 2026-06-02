// i18n.js — bilingual (English / 简体中文) string table and switcher.
// The interface is a reference for hardware clients, so every visible string
// lives here, keyed, with both locales side by side. Default follows the
// browser language, persisted in localStorage.

const STRINGS = {
    en: {
        'app.title': 'DeusRidet',
        'conn.online': 'Connected',
        'conn.offline': 'Connecting…',
        'lang.toggle': '中文',
        'presence.idle': 'Resting',
        'presence.active': 'Awake',
        'presence.daydream': 'Daydreaming',
        'presence.dreaming': 'Dreaming',
        'presence.wakefulness': 'Wakefulness',
        'presence.offline': 'Asleep',
        'listen.idle': 'Listening',
        'listen.hot': 'Hearing speech',
        'listen.mic_on': 'Mic on',
        'listen.mic_off': 'Mic off',
        'speakers.label': 'Voices',
        'speakers.none': 'no one yet',
        'speakers.rename_prompt': 'Name for this speaker',
        'speakers.window_label': 'Speaker window',
        'speakers.window_idle': 'idle',
        'speakers.window_running': 'processing',
        'speakers.window_finalizing': 'finalizing',
        'turn.thinking': 'Thinking',
        'turn.you': 'You',
        'turn.unknown': 'Unknown',
        'turn.speaker_prefix': 'Speaker',
        'turn.online': 'online',
        'turn.prefill': 'prefill',
        'composer.placeholder': 'Say something…',
        'composer.send': 'Send',
        'hint.empty': 'The entity is awake and listening. Speak or type to begin.',
    },
    zh: {
        'app.title': 'DeusRidet',
        'conn.online': '已连接',
        'conn.offline': '连接中…',
        'lang.toggle': 'EN',
        'presence.idle': '静息',
        'presence.active': '清醒',
        'presence.daydream': '走神',
        'presence.dreaming': '做梦',
        'presence.wakefulness': '清醒度',
        'presence.offline': '休眠',
        'listen.idle': '聆听中',
        'listen.hot': '正在听你说话',
        'listen.mic_on': '麦克风开',
        'listen.mic_off': '麦克风关',
        'speakers.label': '说话人',
        'speakers.none': '暂无',
        'speakers.rename_prompt': '给这个说话人起个名字',
        'speakers.window_label': '说话人窗口',
        'speakers.window_idle': '空闲',
        'speakers.window_running': '处理中',
        'speakers.window_finalizing': '收尾中',
        'turn.thinking': '思考',
        'turn.you': '你',
        'turn.unknown': '未知',
        'turn.speaker_prefix': '说话人',
        'turn.online': '在线',
        'turn.prefill': '进入预填充',
        'composer.placeholder': '说点什么…',
        'composer.send': '发送',
        'hint.empty': '智能体已清醒并在聆听。开口说话或打字即可开始。',
    },
};

class I18n {
    constructor() {
        const saved = localStorage.getItem('dr_lang');
        const guess = (navigator.language || 'en').toLowerCase().startsWith('zh') ? 'zh' : 'en';
        this.lang = (saved === 'en' || saved === 'zh') ? saved : guess;
        this._listeners = new Set();
    }

    t(key) {
        return STRINGS[this.lang][key] ?? STRINGS.en[key] ?? key;
    }

    toggle() {
        this.lang = this.lang === 'en' ? 'zh' : 'en';
        localStorage.setItem('dr_lang', this.lang);
        document.documentElement.lang = this.lang === 'zh' ? 'zh-CN' : 'en';
        for (const fn of this._listeners) fn(this.lang);
    }

    onChange(fn) {
        this._listeners.add(fn);
        return () => this._listeners.delete(fn);
    }
}

export const i18n = new I18n();
