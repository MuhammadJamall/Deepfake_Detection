/* ╔══════════════════════════════════════════════════════════╗
   ║  DeepScan — Frontend Logic                              ║
   ║  Handles upload, drag-drop, fetch, and result rendering  ║
   ╚══════════════════════════════════════════════════════════╝ */

(() => {
	"use strict";

	// ── DOM References ──
	const $ = (id) => document.getElementById(id);

	const uploadZone     = $("uploadZone");
	const uploadContent  = $("uploadContent");
	const previewWrap    = $("previewWrap");
	const previewImage   = $("previewImage");
	const imageInput     = $("imageInput");
	const chooseFileBtn  = $("chooseFileBtn");
	const analyzeBtn     = $("analyzeBtn");
	const loadingState   = $("loadingState");
	const resultCard     = $("resultCard");
	const fileMeta       = $("fileMeta");
	const errorAlert     = $("errorAlert");
	const errorMessage   = $("errorMessage");
	const resetBtn       = $("resetBtn");

	const finalVerdict      = $("finalVerdict");
	const confidenceText    = $("confidenceText");
	const confidenceBar     = $("confidenceBar");
	const confidenceBarLabel = $("confidenceBarLabel");
	const verdictRingFill   = $("verdictRingFill");
	const efficientnetValue = $("efficientnetValue");
	const xceptionValue     = $("xceptionValue");

	let selectedFile = null;

	// ── Helpers ──
	const CIRCUMFERENCE = 2 * Math.PI * 54; // ring radius = 54

	function formatSize(bytes) {
		if (bytes < 1024) return bytes + " B";
		if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
		return (bytes / (1024 * 1024)).toFixed(1) + " MB";
	}

	function parsePercent(value) {
		if (typeof value === "number") return Math.max(0, Math.min(100, value));
		const parsed = parseFloat(String(value).replace("%", ""));
		return Number.isNaN(parsed) ? 0 : Math.max(0, Math.min(100, parsed));
	}

	// ── File Selection ──
	function setSelectedFile(file) {
		selectedFile = file || null;
		hideError();

		if (selectedFile) {
			fileMeta.textContent = `${selectedFile.name}  ·  ${formatSize(selectedFile.size)}`;
			analyzeBtn.disabled = false;

			// Show image preview
			const reader = new FileReader();
			reader.onload = (e) => {
				previewImage.src = e.target.result;
				uploadContent.classList.add("d-none");
				previewWrap.classList.remove("d-none");
			};
			reader.readAsDataURL(selectedFile);
		} else {
			fileMeta.textContent = "";
			analyzeBtn.disabled = true;
			previewWrap.classList.add("d-none");
			uploadContent.classList.remove("d-none");
			previewImage.src = "";
		}
	}

	// ── Loading State ──
	function setLoading(isLoading) {
		if (isLoading) {
			loadingState.classList.remove("d-none");
			analyzeBtn.disabled = true;
			chooseFileBtn.disabled = true;
			resultCard.classList.add("d-none");
			hideError();
		} else {
			loadingState.classList.add("d-none");
			analyzeBtn.disabled = !selectedFile;
			chooseFileBtn.disabled = false;
		}
	}

	// ── Error Display ──
	function showError(msg) {
		errorMessage.textContent = msg;
		errorAlert.classList.remove("d-none");
	}

	function hideError() {
		errorAlert.classList.add("d-none");
	}

	// ── Render Results ──
	function updateResultUI(payload) {
		const verdict = String(payload.final_label || "UNKNOWN").toUpperCase();
		const conf = parsePercent(payload.confidence);

		// Verdict label
		finalVerdict.textContent = verdict;
		finalVerdict.classList.remove("verdict-real", "verdict-fake");
		confidenceBar.classList.remove("confidence-real", "confidence-fake");

		const isReal = verdict === "REAL";
		const colorClass = isReal ? "real" : "fake";

		finalVerdict.classList.add(`verdict-${colorClass}`);
		confidenceBar.classList.add(`confidence-${colorClass}`);

		// Ring color
		const ringColor = isReal
			? "var(--success)"
			: "var(--danger)";
		verdictRingFill.style.stroke = ringColor;

		// Ring progress (animate dashoffset)
		const offset = CIRCUMFERENCE - (conf / 100) * CIRCUMFERENCE;
		verdictRingFill.style.strokeDashoffset = offset;

		// Confidence text
		const confLabel = conf.toFixed(1) + "%";
		confidenceText.textContent = confLabel;
		confidenceBarLabel.textContent = confLabel;

		// Confidence bar
		confidenceBar.style.width = conf + "%";

		// Model values
		efficientnetValue.textContent = payload.efficientnet_confidence || "—";
		xceptionValue.textContent = payload.xception_confidence || "—";

		// Show result card with animation
		resultCard.classList.remove("d-none", "result-reveal");
		// Force reflow for animation restart
		void resultCard.offsetWidth;
		resultCard.classList.add("result-reveal");

		// Scroll into view
		resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
	}

	// ── Detection API ──
	async function runDetection() {
		if (!selectedFile) return;

		const formData = new FormData();
		formData.append("image", selectedFile);

		setLoading(true);

		try {
			const response = await fetch("/detect", {
				method: "POST",
				body: formData,
			});

			const payload = await response.json();

			if (!response.ok) {
				throw new Error(payload.error || "Detection failed");
			}

			updateResultUI(payload);
		} catch (error) {
			showError(error.message || "Unable to process image. Please try again.");
		} finally {
			setLoading(false);
		}
	}

	// ── Reset ──
	function resetUI() {
		setSelectedFile(null);
		imageInput.value = "";
		resultCard.classList.add("d-none");
		hideError();

		// Reset ring
		verdictRingFill.style.strokeDashoffset = CIRCUMFERENCE;
		confidenceBar.style.width = "0%";
		confidenceText.textContent = "0%";
		confidenceBarLabel.textContent = "0%";
		finalVerdict.textContent = "—";
		finalVerdict.classList.remove("verdict-real", "verdict-fake");
		confidenceBar.classList.remove("confidence-real", "confidence-fake");
		efficientnetValue.textContent = "—";
		xceptionValue.textContent = "—";

		// Scroll to upload
		uploadZone.scrollIntoView({ behavior: "smooth", block: "center" });
	}

	// ── Event Listeners ──
	chooseFileBtn.addEventListener("click", () => imageInput.click());
	analyzeBtn.addEventListener("click", runDetection);
	resetBtn.addEventListener("click", resetUI);

	uploadZone.addEventListener("click", () => imageInput.click());
	uploadZone.addEventListener("keydown", (e) => {
		if (e.key === "Enter" || e.key === " ") {
			e.preventDefault();
			imageInput.click();
		}
	});

	imageInput.addEventListener("change", () => {
		setSelectedFile(imageInput.files?.[0] || null);
	});

	// ── Drag & Drop ──
	["dragenter", "dragover"].forEach((evt) => {
		uploadZone.addEventListener(evt, (e) => {
			e.preventDefault();
			e.stopPropagation();
			uploadZone.classList.add("dragover");
		});
	});

	["dragleave", "drop"].forEach((evt) => {
		uploadZone.addEventListener(evt, (e) => {
			e.preventDefault();
			e.stopPropagation();
			uploadZone.classList.remove("dragover");
		});
	});

	uploadZone.addEventListener("drop", (e) => {
		const file = e.dataTransfer?.files?.[0];
		if (file && file.type.startsWith("image/")) {
			setSelectedFile(file);
		} else if (file) {
			showError("Please drop an image file (JPG, PNG, WEBP).");
		}
	});

	// Prevent full-page drops
	document.addEventListener("dragover", (e) => e.preventDefault());
	document.addEventListener("drop", (e) => e.preventDefault());
})();
