const videos = [...document.querySelectorAll('video')];

function fmt(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

videos.forEach((video) => {
  video.removeAttribute('controls');
  const shell = document.createElement('div');
  shell.className = 'video-shell';
  video.parentNode.insertBefore(shell, video);
  shell.appendChild(video);

  const controls = document.createElement('div');
  controls.className = 'video-controls';
  controls.innerHTML = `
    <button class="vc-btn vc-play" type="button" aria-label="播放">▶</button>
    <input class="vc-seek" type="range" min="0" max="1000" step="1" value="0" aria-label="拖动进度">
    <span class="vc-time">0:00 / 0:00</span>
    <button class="vc-btn vc-full" type="button" aria-label="全屏">⛶</button>
  `;
  shell.appendChild(controls);

  const play = controls.querySelector('.vc-play');
  const seek = controls.querySelector('.vc-seek');
  const time = controls.querySelector('.vc-time');
  const full = controls.querySelector('.vc-full');

  let dragging = false;

  const duration = () => (Number.isFinite(video.duration) ? video.duration : 0);

  const syncSeek = () => {
    if (dragging) return;
    const d = duration();
    seek.value = d ? String(Math.round((video.currentTime / d) * 1000)) : '0';
  };

  const syncTime = () => {
    time.textContent = `${fmt(video.currentTime)} / ${fmt(duration())}`;
  };

  const syncAll = () => {
    syncSeek();
    syncTime();
    play.textContent = video.paused ? '▶' : 'Ⅱ';
  };

  const toggle = () => {
    if (video.paused) video.play().catch(() => {});
    else video.pause();
  };

  play.addEventListener('click', (event) => {
    event.stopPropagation();
    toggle();
  });

  video.addEventListener('click', toggle);
  video.addEventListener('play', () => {
    videos.forEach((other) => {
      if (other !== video && !other.paused) other.pause();
    });
    syncAll();
  });
  video.addEventListener('pause', syncAll);
  video.addEventListener('timeupdate', () => {
    syncTime();
    syncSeek();
  });
  video.addEventListener('loadedmetadata', syncAll);
  video.addEventListener('seeked', () => {
    syncTime();
    if (!dragging) syncSeek();
  });

  // Mouse/touch dragging: suppress timeupdate overwrites until pointer release.
  seek.addEventListener('pointerdown', () => {
    dragging = true;
  });
  const endDrag = () => {
    dragging = false;
    syncSeek();
    syncTime();
  };
  seek.addEventListener('pointerup', endDrag);
  seek.addEventListener('pointercancel', endDrag);
  seek.addEventListener('change', endDrag);
  seek.addEventListener('input', () => {
    const d = duration();
    if (!d) return;
    const t = (Number(seek.value) / 1000) * d;
    if (Number.isFinite(t)) {
      video.currentTime = t;
      syncTime();
    }
  });

  full.addEventListener('click', (event) => {
    event.stopPropagation();
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else shell.requestFullscreen?.().catch(() => {});
  });

  controls.addEventListener('click', (event) => event.stopPropagation());
  syncAll();
});
