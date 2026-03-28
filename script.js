document.addEventListener('DOMContentLoaded', () => {

    // ── 1. Theme Toggle ────────────────────────────────────────────
    const themeToggleBtn = document.getElementById('theme-toggle');
    const darkIcon  = document.querySelector('.theme-icon-dark');
    const lightIcon = document.querySelector('.theme-icon-light');

    if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
        document.body.classList.add('dark-mode');
        darkIcon.classList.add('hidden');
        lightIcon.classList.remove('hidden');
    }

    themeToggleBtn.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        const isDark = document.body.classList.contains('dark-mode');
        darkIcon.classList.toggle('hidden', isDark);
        lightIcon.classList.toggle('hidden', !isDark);
    });

    // ── 2. Element References ──────────────────────────────────────
    const dropZone             = document.getElementById('drop-zone');
    const fileInput            = document.getElementById('file-input');
    const browseBtn            = document.getElementById('browse-btn');
    const sampleBtns           = document.querySelectorAll('.sample-btn');
    const uploadSection        = document.getElementById('upload-section');
    const uploadContainer      = document.querySelector('.upload-container');
    const loadingState         = document.getElementById('loading-state');
    const resultSection        = document.getElementById('result-section');
    const explainabilitySection = document.getElementById('explainability-section');
    const progressBar          = document.querySelector('.progress-bar');

    // ── 3. Progress Bar ────────────────────────────────────────────
    // FIX BUG 2: definisikan startProgressBar & stopProgressBar
    let progressInterval = null;

    function startProgressBar() {
        let progress = 0;
        progressBar.style.width = '0%';
        progressInterval = setInterval(() => {
            progress += Math.random() * 12;
            if (progress >= 90) progress = 90; // berhenti di 90%, tunggu response
            progressBar.style.width = progress + '%';
        }, 300);
    }

    function stopProgressBar() {
        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        setTimeout(() => { progressBar.style.width = '0%'; }, 400);
    }

    // ── 4. Drag & Drop Events ──────────────────────────────────────
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); });
    });
    ['dragenter', 'dragover'].forEach(evt => {
        dropZone.addEventListener(evt, () => dropZone.classList.add('dragover'));
    });
    ['dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, () => dropZone.classList.remove('dragover'));
    });

    dropZone.addEventListener('drop', e => {
        const files = e.dataTransfer.files;
        if (files.length) handleFile(files[0]);
    });

    browseBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', function () {
        if (this.files.length) handleFile(this.files[0]);
    });

    sampleBtns.forEach(btn => {
        btn.addEventListener('click', () => handleSampleFile(btn.src));
    });

    // ── 5. File Handlers ───────────────────────────────────────────
    // FIX BUG 1: definisikan handleFile & handleSampleFile

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Harap upload file gambar yang valid (JPG, PNG, WEBP).');
            return;
        }
        if (file.size > 5 * 1024 * 1024) {
            alert('Ukuran file maksimal 5MB.');
            return;
        }
        const localUrl = URL.createObjectURL(file);
        analyzeWithAPI(localUrl, file);
    }

    function handleSampleFile(src) {
        analyzeWithAPI(src, null); // null = pakai base64
    }

    // ── 6. API Call ────────────────────────────────────────────────
    // FIX BUG 3: analyzeWithAPI sekarang dipanggil dari handleFile & handleSampleFile

    async function analyzeWithAPI(imageSrc, imageFile = null) {
        uploadContainer.classList.add('hidden');
        loadingState.classList.remove('hidden');
        resultSection.classList.add('hidden');
        explainabilitySection.classList.add('hidden');
        startProgressBar();

        try {
            let response;

            if (imageFile) {
                const formData = new FormData();
                formData.append('image', imageFile);
                response = await fetch('http://localhost:5000/predict', {
                    method: 'POST',
                    body: formData,
                });
            } else {
                const base64 = await urlToBase64(imageSrc);
                response = await fetch('http://localhost:5000/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_base64: base64 }),
                });
            }

            if (!response.ok) throw new Error('Server error: ' + response.status);
            const result = await response.json();

            stopProgressBar();
            setTimeout(() => displayResultFromAPI(imageSrc, result), 400);

        } catch (err) {
            stopProgressBar();
            loadingState.classList.add('hidden');
            uploadContainer.classList.remove('hidden');
            alert('Gagal menghubungi server: ' + err.message +
                  '\nPastikan backend berjalan di http://localhost:5000');
        }
    }

    async function urlToBase64(url) {
        const res  = await fetch(url);
        const blob = await res.blob();
        return new Promise(resolve => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.readAsDataURL(blob);
        });
    }

    // ── 7. Display Result ──────────────────────────────────────────
    // FIX BUG 4: parseFloat() agar aman meski backend kirim string

    function displayResultFromAPI(imageSrc, result) {
        loadingState.classList.add('hidden');
        resultSection.classList.remove('hidden');
        explainabilitySection.classList.remove('hidden');

        document.getElementById('preview-image').src    = imageSrc;
        document.getElementById('explain-original').src = imageSrc;

        // ── Grad-CAM asli dari backend ──────────────────────────────
        if (result.gradcam) {
            document.getElementById('explain-heatmap').src = result.gradcam;
        } else {
            document.getElementById('explain-heatmap').src = imageSrc;
        }

        const confidence      = parseFloat(result.confidence);
        const aiProbability   = parseFloat(result.ai_probability);
        const realProbability = parseFloat(result.real_probability);

        document.getElementById('result-badge').className =
            'result-badge ' + (result.is_ai ? 'ai' : 'real');
        document.getElementById('result-label').textContent = result.label;
        document.getElementById('result-icon').setAttribute('data-lucide',
            result.is_ai ? 'alert-triangle' : 'check-circle');

        document.getElementById('confidence-score').textContent =
            confidence.toFixed(1) + '%';
        document.getElementById('confidence-level').textContent =
            result.confidence_level + ' Confidence';

        document.getElementById('ai-bar').style.width    = aiProbability + '%';
        document.getElementById('real-bar').style.width  = realProbability + '%';
        document.getElementById('ai-prob').textContent   = aiProbability + '%';
        document.getElementById('real-prob').textContent = realProbability + '%';

        lucide.createIcons();
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ── 8. Reset Button ────────────────────────────────────────────
    document.getElementById('reset-btn').addEventListener('click', () => {
        resultSection.classList.add('hidden');
        explainabilitySection.classList.add('hidden');
        uploadContainer.classList.remove('hidden');
        fileInput.value = '';
        uploadSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
});

