(function setInitialTheme(){
  try {
    const persisted = localStorage.getItem('theme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    if (persisted === 'dark' || (!persisted && prefersDark)) {
      document.documentElement.classList.add('dark');
    }
  } catch(e) {}
})();

(function initThemeToggle(){
  function updateIcons(){
    var isDark = document.documentElement.classList.contains('dark');
    var sun = document.getElementById('sunIcon');
    var moon = document.getElementById('moonIcon');
    if (!sun || !moon) return;
    if (isDark) { moon.classList.add('hidden'); sun.classList.remove('hidden'); }
    else { sun.classList.add('hidden'); moon.classList.remove('hidden'); }
  }
  updateIcons();
  var btn = document.getElementById('themeToggle');
  if (btn) {
    btn.addEventListener('click', function(){
      var isDark = document.documentElement.classList.toggle('dark');
      try { localStorage.setItem('theme', isDark ? 'dark' : 'light'); } catch(e) {}
      updateIcons();
    });
  }
})();

// File upload functionality
(function initFileUpload() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const previewWrapper = document.getElementById('previewWrapper');
  const previewImage = document.getElementById('previewImage');
  const deviceTypeSelect = document.getElementById('deviceType');
  const customDeviceWrapper = document.getElementById('customDeviceWrapper');
  const customDeviceInput = document.getElementById('customDevice');

  // Show/hide custom device input based on selection
  if (deviceTypeSelect && customDeviceWrapper) {
    deviceTypeSelect.addEventListener('change', function() {
      if (this.value === 'Other') {
        customDeviceWrapper.classList.remove('hidden');
        customDeviceInput.required = true;
      } else {
        customDeviceWrapper.classList.add('hidden');
        customDeviceInput.required = false;
      }
    });
  }

  // Handle file input change
  if (fileInput) {
    fileInput.addEventListener('change', function(e) {
      handleFileSelect(e.target.files[0]);
    });
  }

  // Handle dropzone click
  if (dropzone) {
    dropzone.addEventListener('click', function() {
      fileInput.click();
    });

    // Handle drag and drop
    dropzone.addEventListener('dragover', function(e) {
      e.preventDefault();
      dropzone.classList.add('border-green-500', 'bg-green-100');
    });

    dropzone.addEventListener('dragleave', function(e) {
      e.preventDefault();
      dropzone.classList.remove('border-green-500', 'bg-green-100');
    });

    dropzone.addEventListener('drop', function(e) {
      e.preventDefault();
      dropzone.classList.remove('border-green-500', 'bg-green-100');
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        handleFileSelect(files[0]);
      }
    });
  }

  function handleFileSelect(file) {
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file.');
      return;
    }

    // Validate file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      alert('File size must be less than 10MB.');
      return;
    }

    // Update file input
    if (fileInput) {
      fileInput.files = new DataTransfer().files;
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      fileInput.files = dataTransfer.files;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
      if (previewImage) {
        previewImage.src = e.target.result;
      }
      if (previewWrapper) {
        previewWrapper.style.display = 'block';
      }
      if (dropzone) {
        dropzone.style.display = 'none';
      }
    };
    reader.readAsDataURL(file);
  }
})();

