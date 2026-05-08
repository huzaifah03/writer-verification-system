// main.js — handles upload previews and form submission

document.addEventListener("DOMContentLoaded", () => {
    setupPreview("assignment1", "preview1");
    setupPreview("assignment2", "preview2");
    setupForm();
});

/**
 * Show an image preview when a file is selected.
 */
function setupPreview(inputId, previewId) {
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);
    if (!input || !preview) return;

    input.addEventListener("change", () => {
        const file = input.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.classList.remove("hidden");
            };
            reader.readAsDataURL(file);
        }
    });
}

/**
 * Handle form submission via fetch (no page reload).
 */
function setupForm() {
    const form = document.getElementById("verifyForm");
    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const resultSection = document.getElementById("resultSection");
        const errorMsg = document.getElementById("errorMsg");
        const spinner = document.getElementById("loadingSpinner");
        const submitBtn = document.getElementById("submitBtn");

        // Reset UI
        resultSection.classList.add("hidden");
        errorMsg.classList.add("hidden");
        spinner.classList.remove("hidden");
        submitBtn.disabled = true;
        submitBtn.textContent = "Analysing…";

        try {
            const formData = new FormData(form);
            const response = await fetch("/verify", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (!response.ok || data.error) {
                showError(data.error || "Something went wrong.");
                return;
            }

            showResult(data);

        } catch (err) {
            showError("Network error. Please check your connection and try again.");
        } finally {
            spinner.classList.add("hidden");
            submitBtn.disabled = false;
            submitBtn.textContent = "🔍 Verify Handwriting";
        }
    });
}

function showResult(data) {
    const resultSection = document.getElementById("resultSection");
    document.getElementById("scorePercent").textContent = data.similarity_percentage.toFixed(1);

    const decisionBadge = document.getElementById("decisionBadge");
    decisionBadge.textContent = data.decision;
    decisionBadge.className = "decision-badge " +
        (data.decision === "Same Writer" ? "badge-same" : "badge-diff");

    const riskBadge = document.getElementById("riskBadge");
    riskBadge.textContent = "Risk Level: " + data.risk_level;
    riskBadge.className = "risk-badge badge-risk-" + data.risk_level.toLowerCase();

    document.getElementById("detailLink").href = "/result/" + data.result_id;

    resultSection.classList.remove("hidden");
    resultSection.scrollIntoView({ behavior: "smooth" });
}

function showError(message) {
    const errorMsg = document.getElementById("errorMsg");
    errorMsg.textContent = "❌ " + message;
    errorMsg.classList.remove("hidden");
}
