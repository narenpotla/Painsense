// =========================================================
// PAINSENSE DASHBOARD
// Live state (/api/state) + history (/api/history)
//
// The score is expression intensity on a synthetic morph scale
// (0-10), not a clinical pain measurement.
// =========================================================

const SETTINGS = Object.assign(
    { historyPoints: 120, staleSec: 3 },
    window.PAINSENSE || {}
);

let chart = null;


// =========================================================
// SMALL HELPERS
// =========================================================

function setText(id, text) {

    const element = document.getElementById(id);

    if (element) {
        element.textContent = text;
    }
}


function hasValue(value) {

    return value !== null && value !== undefined;
}


function historyScore(item) {

    return hasValue(item.score) ? item.score : item.probability;
}


function formatTime(epochSeconds) {

    if (!hasValue(epochSeconds)) {
        return "—";
    }

    return new Date(Number(epochSeconds) * 1000).toLocaleTimeString();
}


// =========================================================
// LIVE / OFFLINE INDICATORS
// =========================================================

function updateCapture(live, sequenceCurrent, sequenceTotal) {

    const badge = document.getElementById("captureBadge");
    const status = document.getElementById("captureStatus");

    if (badge) {
        badge.classList.toggle("offline", !live);
    }

    if (status) {
        status.classList.toggle("offline", !live);
        status.classList.toggle("active", live);
    }

    if (!live) {
        setText("captureHeader", "WAITING FOR CAMERA");
        setText("captureText", "OFFLINE");
        setText("captureDescription", "live_camera.py is not running");
        return;
    }

    setText("captureHeader", "CAPTURE ACTIVE");
    setText("captureText", "ACTIVE");

    if (Number(sequenceCurrent) === 0) {
        setText("captureDescription", "Waiting for a face");
    }
    else if (Number(sequenceCurrent) < Number(sequenceTotal)) {
        setText(
            "captureDescription",
            `Collecting face frames ${sequenceCurrent} / ${sequenceTotal}`
        );
    }
    else {
        setText("captureDescription", "Face detected, scoring");
    }
}


// =========================================================
// SCORE / PREDICTION / ALERT
// =========================================================

function updateScore(score, alert) {

    if (!hasValue(score)) {
        setText("painScore", "—");
        setText("painStatus", "Waiting for a score");
        setText("painProbability", "—");
        return;
    }

    score = Number(score);

    setText("painScore", score.toFixed(1));
    setText("painProbability", score.toFixed(1) + " / 10");

    // The status follows the alert state machine (median + hysteresis),
    // not a single-frame threshold.
    setText(
        "painStatus",
        alert ? "Alert: sustained high intensity" : "Below alert level"
    );
}


function updateAlert(alert, reason) {

    const alertElement = document.getElementById("alertStatus");

    if (!alertElement) {
        return;
    }

    if (alert) {
        alertElement.textContent = "ALERT";
        alertElement.style.color = "#c62828";
        setText("alertReason", reason || "Alert triggered");
    }
    else {
        alertElement.textContent = "STANDBY";
        alertElement.style.color = "#16803c";
        setText("alertReason", "No active alert");
    }
}


// =========================================================
// CHART (optional: skipped if Chart.js did not load)
// =========================================================

function createChart() {

    const canvas = document.getElementById("painChart");

    if (!canvas) {
        return;
    }

    if (typeof Chart === "undefined") {
        console.warn(
            "Chart.js not loaded (offline?). Trend chart disabled; "
            + "cards and table still update."
        );
        return;
    }

    chart = new Chart(canvas.getContext("2d"), {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "Expression intensity",
                data: [],
                tension: 0.3,
                borderWidth: 2,
                pointRadius: 0,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                y: {
                    min: 0,
                    max: 10,
                    title: { display: true, text: "Intensity (0-10)" }
                },
                x: {
                    title: { display: true, text: "Time" },
                    ticks: { maxTicksLimit: 8 }
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}


function updateChart(history) {

    if (!chart) {
        return;
    }

    const points = history
        .filter(function (item) { return hasValue(historyScore(item)); })
        .slice(-SETTINGS.historyPoints);

    chart.data.labels = points.map(function (item) {
        return formatTime(item.timestamp);
    });

    chart.data.datasets[0].data = points.map(function (item) {
        return Number(historyScore(item));
    });

    chart.update();
}


// =========================================================
// RECENT OBSERVATIONS TABLE
// =========================================================

function updateHistoryTable(history) {

    const tableBody = document.getElementById("historyTable");

    if (!tableBody) {
        return;
    }

    tableBody.replaceChildren();

    if (history.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 4;
        cell.className = "empty-state";
        cell.textContent = "No observations recorded yet.";
        row.appendChild(cell);
        tableBody.appendChild(row);
        return;
    }

    history
        .slice()
        .reverse()
        .slice(0, 10)
        .forEach(function (item) {

            const score = historyScore(item);
            const values = [
                formatTime(item.timestamp),
                hasValue(score) ? Number(score).toFixed(1) : "—",
                item.prediction || "—",
                item.alert ? "ALERT" : "—"
            ];

            const row = document.createElement("tr");

            values.forEach(function (value) {
                const cell = document.createElement("td");
                cell.textContent = value;
                row.appendChild(cell);
            });

            tableBody.appendChild(row);
        });
}


// =========================================================
// POLLING
// =========================================================

async function fetchState() {

    try {
        const response = await fetch("/api/state", { cache: "no-store" });

        if (!response.ok) {
            throw new Error("State API request failed");
        }

        const data = await response.json();
        const live = Boolean(data.live);

        updateCapture(live, data.sequence_current, data.sequence_total);
        updateScore(live ? data.score : null, live && data.alert);
        setText("prediction", live && data.prediction ? data.prediction : "—");
        setText(
            "sequenceStatus",
            `${live ? data.sequence_current : 0} / ${data.sequence_total}`
        );
        setText("updatedAt", data.updated_at || "—");
        updateAlert(live && data.alert, data.alert_reason);
    }
    catch (error) {
        console.error("PainSense state API error:", error);
        updateCapture(false, 0, 0);
    }
}


async function fetchHistory() {

    try {
        const response = await fetch(
            `/api/history?n=${SETTINGS.historyPoints}`,
            { cache: "no-store" }
        );

        if (!response.ok) {
            throw new Error("History API request failed");
        }

        const history = await response.json();

        if (!Array.isArray(history)) {
            return;
        }

        updateHistoryTable(history);
        updateChart(history);
    }
    catch (error) {
        console.error("PainSense history API error:", error);
    }
}


// =========================================================
// START
// =========================================================

document.addEventListener("DOMContentLoaded", function () {

    // A chart failure must never stop polling.
    try {
        createChart();
    }
    catch (error) {
        console.error("Chart setup failed:", error);
        chart = null;
    }

    fetchState();
    fetchHistory();

    setInterval(fetchState, 1000);
    setInterval(fetchHistory, 2000);
});
