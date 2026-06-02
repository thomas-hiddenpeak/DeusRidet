// event-bus.js — minimal pub/sub for inter-component messaging.
// Components never reference each other directly; they speak through the bus.

export class EventBus {
    constructor() {
        this._handlers = new Map();   // event name -> Set<fn>
    }

    on(event, fn) {
        if (!this._handlers.has(event)) this._handlers.set(event, new Set());
        this._handlers.get(event).add(fn);
        return () => this.off(event, fn);
    }

    off(event, fn) {
        this._handlers.get(event)?.delete(fn);
    }

    emit(event, payload) {
        const set = this._handlers.get(event);
        if (!set) return;
        for (const fn of set) {
            try { fn(payload); }
            catch (e) { console.error(`[bus] handler for "${event}" threw`, e); }
        }
    }
}
