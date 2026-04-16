const uploadZone = document.getElementById("uploadZone");
const imageInput = document.getElementById("imageInput");
const chooseFileBtn = document.getElementById("chooseFileBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const loadingState = document.getElementById("loadingState");
const resultCard = document.getElementById("resultCard");
const fileMeta = document.getElementById("fileMeta");

const finalVerdict = document.getElementById("finalVerdict");
const confidenceText = document.getElementById("confidenceText");
const confidenceBar = document.getElementById("confidenceBar");
const efficientnetValue = document.getElementById("efficientnetValue");
const xceptionValue = document.getElementById("xceptionValue");

let selectedFile = null;

function setSelectedFile(file) {
	selectedFile = file || null;

	if (selectedFile) {
		fileMeta.textContent = "Selected: " + selectedFile.name;
		analyzeBtn.disabled = false;
	} else {
		fileMeta.textContent = "";
		analyzeBtn.disabled = true;
	}
}

function parsePercent(value) {
	if (typeof value === "number") {
		return Math.max(0, Math.min(100, value));
	}

	const parsed = parseFloat(String(value).replace("%", ""));
	if (Number.isNaN(parsed)) {
		return 0;
	}
	return Math.max(0, Math.min(100, parsed));
}

function setLoading(isLoading) {
	if (isLoading) {
		loadingState.classList.remove("d-none");
		analyzeBtn.disabled = true;
	} else {
		loadingState.classList.add("d-none");
		analyzeBtn.disabled = !selectedFile;
	}
}

function updateResultUI(payload) {
	const verdict = String(payload.final_label || "UNKNOWN").toUpperCase();
	const conf = parsePercent(payload.confidence);

	finalVerdict.textContent = verdict;
	finalVerdict.classList.remove("verdict-real", "verdict-fake");
	confidenceBar.classList.remove("confidence-real", "confidence-fake");

	if (verdict === "REAL") {
		finalVerdict.classList.add("verdict-real");
		confidenceBar.classList.add("confidence-real");
	} else if (verdict === "FAKE") {
		finalVerdict.classList.add("verdict-fake");
		confidenceBar.classList.add("confidence-fake");
	}

	const confidenceLabel = conf.toFixed(2) + "%";
	confidenceText.textContent = confidenceLabel;
	confidenceBar.style.width = confidenceLabel;
	confidenceBar.textContent = confidenceLabel;
	confidenceBar.parentElement.setAttribute("aria-valuenow", conf.toFixed(2));

	efficientnetValue.textContent = payload.efficientnet_confidence || payload.efficientnet || "-";
	xceptionValue.textContent = payload.xception_confidence || payload.xception || "-";

	resultCard.classList.remove("d-none");
}

async function runDetection() {
	if (!selectedFile) {
		return;
	}

	const formData = new FormData();
	formData.append("image", selectedFile);

	setLoading(true);

	try {
		const response = await fetch("/detect", {
			method: "POST",
			body: formData
		});

		const payload = await response.json();
		if (!response.ok) {
			throw new Error(payload.error || "Detection failed");
		}

		updateResultUI(payload);
	} catch (error) {
		alert(error.message || "Unable to process image.");
	} finally {
		setLoading(false);
	}
}

chooseFileBtn.addEventListener("click", () => imageInput.click());
analyzeBtn.addEventListener("click", runDetection);

uploadZone.addEventListener("click", () => imageInput.click());
uploadZone.addEventListener("keydown", (event) => {
	if (event.key === "Enter" || event.key === " ") {
		event.preventDefault();
		imageInput.click();
	}
});

imageInput.addEventListener("change", () => {
	setSelectedFile(imageInput.files && imageInput.files[0] ? imageInput.files[0] : null);
});

["dragenter", "dragover"].forEach((eventName) => {
	uploadZone.addEventListener(eventName, (event) => {
		event.preventDefault();
		event.stopPropagation();
		uploadZone.classList.add("dragover");
	});
});

["dragleave", "drop"].forEach((eventName) => {
	uploadZone.addEventListener(eventName, (event) => {
		event.preventDefault();
		event.stopPropagation();
		uploadZone.classList.remove("dragover");
	});
});

uploadZone.addEventListener("drop", (event) => {
	const file = event.dataTransfer && event.dataTransfer.files ? event.dataTransfer.files[0] : null;
	if (file) {
		setSelectedFile(file);
	}
});
