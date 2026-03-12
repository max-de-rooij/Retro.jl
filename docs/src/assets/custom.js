// ── Retro.jl docs — folder-tab sidebar & card animations ──────────

document.addEventListener("DOMContentLoaded", () => {
    // ── 1. Create the folder tab ──────────────────────────────────
    const tab = document.createElement("div");
    tab.id = "retro-folder-tab";
    tab.innerHTML = '<span class="tab-label">Menu</span>';

    const overlay = document.createElement("div");
    overlay.id = "retro-sidebar-overlay";

    document.body.appendChild(tab);
    document.body.appendChild(overlay);

    const sidebar = document.querySelector(".docs-sidebar");

    function openMenu() {
        sidebar?.classList.add("retro-open");
        tab.classList.add("open");
        overlay.classList.add("active");
    }

    function closeMenu() {
        sidebar?.classList.remove("retro-open");
        tab.classList.remove("open");
        overlay.classList.remove("active");
    }

    tab.addEventListener("click", () => {
        if (tab.classList.contains("open")) {
            closeMenu();
        } else {
            openMenu();
        }
    });

    overlay.addEventListener("click", closeMenu);

    // Close on Escape
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeMenu();
    });

    // ── 1b. Inject home link into navbar ──────────────────────────
    const navbar = document.querySelector(".docs-navbar");
    if (navbar) {
        const homeLink = document.createElement("a");
        homeLink.id = "retro-home-link";
        homeLink.href = typeof documenterBaseURL !== "undefined"
            ? documenterBaseURL + "/index.html"
            : "/index.html";
        homeLink.textContent = "RETRO.JL";
        navbar.insertBefore(homeLink, navbar.firstChild);

        // ── 1c. Replace right side with GitHub + theme toggle ─────
        const docsRight = navbar.querySelector(".docs-right");
        if (docsRight) docsRight.style.display = "none";

        const rightGroup = document.createElement("div");
        rightGroup.id = "retro-navbar-right";

        // GitHub link
        const ghLink = document.createElement("a");
        ghLink.id = "retro-github-link";
        ghLink.href = "https://github.com/max-de-rooij/Retro.jl";
        ghLink.target = "_blank";
        ghLink.rel = "noopener";
        ghLink.innerHTML =
            '<svg viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.65 7.65 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>' +
            '<span>GitHub</span>';
        rightGroup.appendChild(ghLink);

        // Theme toggle
        const toggle = document.createElement("div");
        toggle.id = "retro-theme-toggle";
        toggle.title = "Toggle dark / light mode";
        toggle.innerHTML = '<div class="toggle-light"></div><div class="rocker"><div class="rocker-top">I</div><div class="rocker-bottom">O</div></div>';

        const savedTheme = localStorage.getItem("retro-theme");
        if (savedTheme === "light") {
            document.body.classList.add("retro-light");
            toggle.classList.add("light");
        }

        toggle.addEventListener("click", () => {
            const isLight = document.body.classList.toggle("retro-light");
            toggle.classList.toggle("light", isLight);
            localStorage.setItem("retro-theme", isLight ? "light" : "dark");
        });

        rightGroup.appendChild(toggle);
        navbar.appendChild(rightGroup);
    }

    // ── 2. Animate "I want to:" cards on scroll ──────────────────
    const cards = document.querySelectorAll(".retro-action-card");
    if (cards.length) {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add("retro-visible");
                        observer.unobserve(entry.target);
                    }
                });
            },
            { threshold: 0.15 }
        );
        cards.forEach((card) => observer.observe(card));
    }

    // ── 3. Retro admonition icons ────────────────────────────────
    document.querySelectorAll(".admonition-title").forEach((title) => {
        // Strip leading emoji (any emoji / symbol codepoints + trailing space)
        const text = title.textContent.replace(/^[\p{Emoji}\p{So}\u200d\ufe0f]+\s*/u, "").trim();
        title.textContent = "";

        // Inject pixel-art icon
        const icon = document.createElement("span");
        icon.className = "retro-admonition-icon";
        icon.setAttribute("aria-hidden", "true");
        title.appendChild(icon);

        // Re-add cleaned text
        const label = document.createTextNode(text);
        title.appendChild(label);
    });
});