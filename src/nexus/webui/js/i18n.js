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
        'speakers.window_cycle': 'Next periodic re-ID',
        'speakers.window_idle': 'idle',
        'speakers.window_running': 'processing',
        'speakers.window_finalizing': 'finalizing',
        'speakers.window_disabled': 'disabled',
        'contacts.title': 'People you know',
        'contacts.known': 'Named',
        'contacts.unknown': 'Unlabeled',
        'contacts.meta_named': 'labeled',
        'contacts.meta_unknown': 'unlabeled',
        'contacts.link_candidate': 'candidate',
        'contacts.link_stable': 'stable',
        'contacts.empty_known': 'No labeled people yet',
        'contacts.empty_unknown': 'No unlabeled speakers detected',
        'contacts.now_speaking': 'Now speaking',
        'contacts.no_active': 'No active speaker',
        'contacts.remove': 'Remove from list',
        'contacts.rename': 'Rename',
        'contacts.merge': 'Merge into another contact',
        'contacts.merge_none': 'No other contact to merge into yet.',
        'contacts.copy_id': 'Copy ID',
        'contacts.copied': 'ID copied',
        'contacts.action_name': 'Name',
        'contacts.action_merge': 'Merge',
        'contacts.name_placeholder': 'Enter a display name',
        'contacts.save': 'Save',
        'contacts.cancel': 'Cancel',
        'turn.thinking': 'Thinking',
        'turn.you': 'You',
        'turn.unknown': 'Unknown',
        'turn.speaker_prefix': 'Speaker',
        'turn.asr_live': 'Live ASR',
        'turn.online': 'online',
        'turn.prefill': 'prefill',
        'turn.lane_live': 'Live ASR (editable)',
        'turn.lane_prefill': 'Prefill committed (immutable)',
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
        'speakers.window_cycle': '下一轮周期重识别',
        'speakers.window_idle': '空闲',
        'speakers.window_running': '处理中',
        'speakers.window_finalizing': '收尾中',
        'speakers.window_disabled': '关闭',
        'contacts.title': '认识的人',
        'contacts.known': '已标注',
        'contacts.unknown': '未标注',
        'contacts.meta_named': '已命名',
        'contacts.meta_unknown': '待命名',
        'contacts.link_candidate': '候选',
        'contacts.link_stable': '稳定',
        'contacts.empty_known': '暂无已标注的人',
        'contacts.empty_unknown': '暂无未标注说话人',
        'contacts.now_speaking': '当前说话人',
        'contacts.no_active': '暂无正在说话的人',
        'contacts.remove': '从列表删除',
        'contacts.rename': '改名',
        'contacts.merge': '并入其他人',
        'contacts.merge_none': '暂无其他联系人可供合并。',
        'contacts.copy_id': '复制 ID',
        'contacts.copied': '已复制 ID',
        'contacts.action_name': '命名',
        'contacts.action_merge': '并入',
        'contacts.name_placeholder': '输入显示名称',
        'contacts.save': '保存',
        'contacts.cancel': '取消',
        'turn.thinking': '思考',
        'turn.you': '你',
        'turn.unknown': '未知',
        'turn.speaker_prefix': '说话人',
        'turn.asr_live': '实时ASR',
        'turn.online': '在线',
        'turn.prefill': '进入预填充',
        'turn.lane_live': '实时ASR（可修正）',
        'turn.lane_prefill': '进入Prefill（不可修改）',
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
