// speaker-colors.js — Shared speaker color palette for consistent visuals.

export const SPK_PALETTE = [
    '#58a6ff', '#3fb950', '#d29922', '#f85149',
    '#bc8cff', '#79c0ff', '#56d364', '#e3b341',
    '#ff7b72', '#d2a8ff', '#f778ba', '#a5d6ff'
];

export const SPK_UNKNOWN_COLOR = '#484f58';  // gray for id=-1 (identifying...)

export function spkColor(id) {
    if (id < 0) return SPK_UNKNOWN_COLOR;
    return SPK_PALETTE[id % SPK_PALETTE.length];
}
