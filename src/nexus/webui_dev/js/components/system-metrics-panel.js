// system-metrics-panel.js — System resource monitoring panel
//
// Displays: CUDA memory, system memory (MemAvail), process RSS,
// and KV cache block utilization with visual bars.

export class SystemMetricsPanel {
    constructor(ws) {
        this.ws = ws;
        this.el = document.getElementById('system-metrics-panel');
        if (!this.el) return;

        this.cudaFreeMb = 0;
        this.cudaTotalMb = 0;
        this.memAvailMb = 0;
        this.rssMb = 0;
        this.kvUsed = 0;
        this.kvFree = 0;

        this._render();
    }

    _render() {
        this.el.innerHTML = `
            <h2 class="panel__title">System Metrics</h2>
            <div class="sm-row">
                <span class="sm-label">CUDA Memory</span>
                <div class="sm-bar-wrap">
                    <div class="sm-bar sm-bar--cuda" id="sm-cuda-bar"></div>
                </div>
                <span class="sm-val" id="sm-cuda-val">— / — GB</span>
            </div>
            <div class="sm-row">
                <span class="sm-label">System Avail</span>
                <div class="sm-bar-wrap">
                    <div class="sm-bar sm-bar--mem" id="sm-mem-bar"></div>
                </div>
                <span class="sm-val" id="sm-mem-val">— GB</span>
            </div>
            <div class="sm-row">
                <span class="sm-label">Process RSS</span>
                <div class="sm-bar-wrap">
                    <div class="sm-bar sm-bar--rss" id="sm-rss-bar"></div>
                </div>
                <span class="sm-val" id="sm-rss-val">— GB</span>
            </div>
            <div class="sm-row">
                <span class="sm-label">KV Blocks</span>
                <div class="sm-bar-wrap">
                    <div class="sm-bar sm-bar--kv" id="sm-kv-bar"></div>
                </div>
                <span class="sm-val" id="sm-kv-val">0 / 0</span>
            </div>
        `;

        this.cudaBar = this.el.querySelector('#sm-cuda-bar');
        this.cudaVal = this.el.querySelector('#sm-cuda-val');
        this.memBar = this.el.querySelector('#sm-mem-bar');
        this.memVal = this.el.querySelector('#sm-mem-val');
        this.rssBar = this.el.querySelector('#sm-rss-bar');
        this.rssVal = this.el.querySelector('#sm-rss-val');
        this.kvBar = this.el.querySelector('#sm-kv-bar');
        this.kvVal = this.el.querySelector('#sm-kv-val');
    }

    onConsciousnessState(data) {
        if (!this.el) return;

        if (data.cuda_free_mb !== undefined && data.cuda_total_mb !== undefined) {
            this.cudaFreeMb = data.cuda_free_mb;
            this.cudaTotalMb = data.cuda_total_mb;
            const used = this.cudaTotalMb - this.cudaFreeMb;
            const pct = this.cudaTotalMb > 0 ? (used / this.cudaTotalMb) * 100 : 0;
            this.cudaBar.style.width = Math.min(100, pct) + '%';
            this.cudaBar.className = 'sm-bar sm-bar--cuda' + (pct > 90 ? ' sm-bar--danger' : pct > 75 ? ' sm-bar--warn' : '');
            this.cudaVal.textContent = `${(used / 1024).toFixed(1)} / ${(this.cudaTotalMb / 1024).toFixed(1)} GB`;
        }

        if (data.mem_avail_mb !== undefined) {
            this.memAvailMb = data.mem_avail_mb;
            // Orin 64 GB total system memory
            const totalMb = 65536;
            const usedPct = ((totalMb - this.memAvailMb) / totalMb) * 100;
            this.memBar.style.width = Math.min(100, usedPct) + '%';
            this.memBar.className = 'sm-bar sm-bar--mem' + (usedPct > 90 ? ' sm-bar--danger' : usedPct > 75 ? ' sm-bar--warn' : '');
            this.memVal.textContent = `${(this.memAvailMb / 1024).toFixed(1)} GB free`;
        }

        if (data.rss_mb !== undefined) {
            this.rssMb = data.rss_mb;
            const rssPct = Math.min(100, (this.rssMb / 65536) * 100);
            this.rssBar.style.width = rssPct + '%';
            this.rssVal.textContent = `${(this.rssMb / 1024).toFixed(1)} GB`;
        }

        if (data.kv_used !== undefined && data.kv_free !== undefined) {
            this.kvUsed = data.kv_used;
            this.kvFree = data.kv_free;
            const total = this.kvUsed + this.kvFree;
            const pct = total > 0 ? (this.kvUsed / total) * 100 : 0;
            this.kvBar.style.width = Math.min(100, pct) + '%';
            this.kvBar.className = 'sm-bar sm-bar--kv' + (pct > 90 ? ' sm-bar--danger' : pct > 75 ? ' sm-bar--warn' : '');
            this.kvVal.textContent = `${this.kvUsed} / ${total}`;
        }
    }
}
