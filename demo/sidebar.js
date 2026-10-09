() => {
    const install = () => {
        const sidebar = document.getElementById('sidebar');
        if (!sidebar) return false;
        if (sidebar.querySelector('.sidebar-resizer')) return true;
        const shell = sidebar.closest('.sidebar-parent');
        if (!shell) return false;
        const key = 'deepseek-sidebar-width';
        const handle = document.createElement('div');
        handle.className = 'sidebar-resizer';
        handle.tabIndex = 0;
        handle.setAttribute('role', 'separator');
        handle.setAttribute('aria-orientation', 'vertical');
        handle.setAttribute('aria-label', 'Cambiar ancho de la barra lateral');
        handle.title = 'Arrastrá para cambiar el ancho. Doble clic para restablecer.';
        sidebar.appendChild(handle);
        let preferred = 320;
        try {
            const saved = Number(localStorage.getItem(key));
            if (saved >= 240 && saved <= 640) preferred = saved;
        } catch (_) { /* Storage may be disabled. Resizing still works. */ }
        const maximum = () => Math.max(240, Math.min(640, window.innerWidth - 360));
        const apply = () => {
            const width = Math.max(240, Math.min(maximum(), preferred));
            shell.style.setProperty('--user-sidebar-width', `${width}px`);
            // Gradio scopes custom CSS beneath .contain, but this shell is its
            // parent. Set its offset directly so it follows the resized panel.
            const open = sidebar.classList.contains('open') && window.innerWidth > 768;
            shell.style.setProperty('padding-left', open ? `${width}px` : '0px', 'important');
            shell.style.setProperty('box-sizing', 'border-box');
            handle.setAttribute('aria-valuemin', '240');
            handle.setAttribute('aria-valuemax', String(maximum()));
            handle.setAttribute('aria-valuenow', String(Math.round(width)));
        };
        const save = () => {
            try { localStorage.setItem(key, String(preferred)); } catch (_) {}
        };
        let pointer = null;
        let startX = 0;
        let startWidth = 0;
        handle.addEventListener('pointerdown', event => {
            if (event.button !== 0 || window.innerWidth <= 768) return;
            event.preventDefault();
            pointer = event.pointerId;
            startX = event.clientX;
            startWidth = sidebar.getBoundingClientRect().width;
            handle.setPointerCapture(pointer);
            shell.classList.add('sidebar-resizing');
        });
        handle.addEventListener('pointermove', event => {
            if (event.pointerId !== pointer) return;
            preferred = Math.max(240, Math.min(maximum(), startWidth + event.clientX - startX));
            apply();
        });
        const finish = () => {
            pointer = null;
            shell.classList.remove('sidebar-resizing');
            save();
        };
        handle.addEventListener('pointerup', finish);
        handle.addEventListener('pointercancel', finish);
        handle.addEventListener('lostpointercapture', finish);
        handle.addEventListener('dblclick', () => { preferred = 320; apply(); save(); });
        handle.addEventListener('keydown', event => {
            const current = Number(handle.getAttribute('aria-valuenow'));
            if (event.key === 'ArrowLeft') preferred = current - 16;
            else if (event.key === 'ArrowRight') preferred = current + 16;
            else if (event.key === 'Home') preferred = 240;
            else if (event.key === 'End') preferred = maximum();
            else return;
            event.preventDefault();
            preferred = Math.max(240, Math.min(maximum(), preferred));
            apply();
            save();
        });
        new MutationObserver(apply).observe(sidebar, { attributes: true, attributeFilter: ['class'] });
        window.addEventListener('resize', apply);
        apply();
        return true;
    };
    if (!install()) {
        const observer = new MutationObserver(() => {
            if (install()) observer.disconnect();
        });
        observer.observe(document.body, { childList: true, subtree: true });
        setTimeout(() => observer.disconnect(), 15000);
    }
}
