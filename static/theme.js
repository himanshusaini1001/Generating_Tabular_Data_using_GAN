(function () {
  const body = document.body;
  const savedTheme = localStorage.getItem('stackedSeqganTheme');
  if (savedTheme === 'light') {
    body.classList.add('light-theme');
  }

  const toggleBtn = document.getElementById('themeToggle');

  const setLabel = () => {
    if (!toggleBtn) return;
    const isLight = body.classList.contains('light-theme');
    toggleBtn.textContent = isLight ? '🌞 Light Mode' : '🌙 Dark Mode';
    toggleBtn.setAttribute('aria-pressed', String(isLight));
  };

  setLabel();

  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      body.classList.toggle('light-theme');
      const nextTheme = body.classList.contains('light-theme') ? 'light' : 'dark';
      localStorage.setItem('stackedSeqganTheme', nextTheme);
      setLabel();
    });
  }
})();

