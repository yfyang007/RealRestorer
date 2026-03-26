(() => {
  const rankingData = Array.isArray(window.rankingData) ? [...window.rankingData] : [];
  const showcaseData = Array.isArray(window.showcaseData) ? [...window.showcaseData] : [];
  const metricLabel = window.rankingMetricLabel || "Final Score";

  const sampleState = Object.fromEntries(showcaseData.map((task) => [task.id, 0]));
  const compareState = Object.fromEntries(showcaseData.map((task) => [task.id, 50]));
  let activeTaskIndex = 0;
  let lastTaskNavDirection = "next";
  let hasRenderedResults = false;

  const q = (selector) => document.querySelector(selector);

  function getOptimizedAssetPath(assetPath, variant = "full") {
    const suffixByVariant = {
      full: ".opt.webp",
      thumb: ".thumb.webp",
      logo: ".logo.webp"
    };

    return assetPath.replace(/(\.[^./]+)$/u, suffixByVariant[variant] || suffixByVariant.full);
  }

  function renderImage({
    src,
    alt,
    className = "",
    loading = "lazy",
    fetchPriority = "auto",
    variant = "full"
  }) {
    const optimizedSrc = getOptimizedAssetPath(src, variant);
    const classAttribute = className ? ` class="${className}"` : "";

    return `
      <img${classAttribute} src="${optimizedSrc}" alt="${alt}" loading="${loading}" decoding="async" fetchpriority="${fetchPriority}">
    `;
  }

  function formatScore(score) {
    return Number(score).toFixed(3);
  }

  function formatTaskPosition(position) {
    return String(position).padStart(2, "0");
  }

  function getWrappedTaskIndex(index) {
    if (showcaseData.length === 0) {
      return 0;
    }

    return (index + showcaseData.length) % showcaseData.length;
  }

  function getSortedRanking() {
    const closedSource = rankingData
      .filter((entry) => !entry.openSource)
      .sort((left, right) => right.score - left.score);

    const openSource = rankingData
      .filter((entry) => entry.openSource && !entry.highlight)
      .sort((left, right) => right.score - left.score);

    const ours = rankingData.filter((entry) => entry.highlight);

    return [...closedSource, ...openSource, ...ours];
  }

  function renderRanking(sortedRanking) {
    const rankingRoot = q("[data-ranking-root]");
    const metricRoots = document.querySelectorAll("[data-metric-label], [data-footer-metric-label]");

    metricRoots.forEach((node) => {
      node.textContent = metricLabel;
    });

    if (!rankingRoot || sortedRanking.length === 0) {
      return;
    }

    const maxScore = Math.max(...sortedRanking.map((entry) => entry.score));
    const chartMax = Math.ceil(maxScore * 1000 / 20) * 20 / 1000;
    const tickCount = 5;
    const ticks = Array.from({ length: tickCount }, (_, index) => {
      const value = chartMax - (chartMax / (tickCount - 1)) * index;
      const bottom = (index / (tickCount - 1)) * 100;
      return {
        label: value.toFixed(3),
        bottom
      };
    });

    rankingRoot.innerHTML = `
      <div class="ranking-chart">
        <div class="chart-axis">
          ${ticks
            .map(
              (tick) => `
                <span class="axis-label" style="bottom:${tick.bottom}%">${tick.label}</span>
              `
            )
            .join("")}
        </div>
        <div class="chart-plot">
          <div class="plot-grid">
            ${ticks
              .map(
                (tick) => `
                  <span class="grid-line" style="bottom:${tick.bottom}%"></span>
                `
              )
              .join("")}
          </div>
          <div class="bar-list">
            ${sortedRanking
              .map((entry) => {
                const height = ((entry.score / chartMax) * 100).toFixed(2);
                const cardClass = entry.highlight ? "bar-card is-highlight" : "bar-card";
                const logoWrapClass = entry.highlight ? "bar-logo-wrap is-highlight" : "bar-logo-wrap";
                const logoClass = entry.highlight ? "bar-logo is-highlight-logo" : "bar-logo";

                return `
                  <article class="${cardClass}" style="--accent:${entry.accent}">
                    <span class="bar-value">${formatScore(entry.score)}</span>
                    <div class="bar-rail">
                      <div class="bar-fill" data-height="${height}%"></div>
                    </div>
                    <div class="${logoWrapClass}">
                      ${renderImage({
                        className: logoClass,
                        src: entry.logo,
                        alt: `${entry.model} logo`,
                        loading: "lazy",
                        fetchPriority: "low",
                        variant: "logo"
                      })}
                    </div>
                    <span class="bar-name">${entry.model}</span>
                  </article>
                `;
              })
              .join("")}
          </div>
        </div>
      </div>
    `;

    const fills = rankingRoot.querySelectorAll(".bar-fill");
    requestAnimationFrame(() => {
      fills.forEach((fill, index) => {
        window.setTimeout(() => {
          fill.style.height = fill.dataset.height;
        }, 70 * index);
      });
    });
  }

  function renderResults({ animateTaskChange = false } = {}) {
    const resultsRoot = q("[data-results-root]");
    if (!resultsRoot || showcaseData.length === 0) {
      return;
    }

    hasRenderedResults = true;

    const task = showcaseData[activeTaskIndex];
    const activeSampleIndex = Math.min(sampleState[task.id] || 0, task.samples.length - 1);
    const activeSample = task.samples[activeSampleIndex];
    const compareSplit = compareState[task.id] ?? 50;
    const transitionClass = animateTaskChange
      ? ` showcase-carousel is-animating is-from-${lastTaskNavDirection}`
      : " showcase-carousel";

    resultsRoot.innerHTML = `
      <div class="${transitionClass.trim()}" style="--showcase-accent:${task.accent}">
        <div class="showcase-nav-meta">
          <span class="showcase-nav-count">${formatTaskPosition(activeTaskIndex + 1)} / ${formatTaskPosition(showcaseData.length)}</span>
          <span class="showcase-nav-name">${task.title}</span>
        </div>

        <div class="showcase-stage" aria-label="Case group navigation">
          <button class="showcase-side-btn showcase-side-btn-prev" type="button" data-task-nav="prev" aria-label="Show previous case group">
            <span class="nav-btn-icon" aria-hidden="true">←</span>
          </button>

          <section class="showcase-row">
            <h3 class="showcase-title">${task.title}</h3>

            <div class="compare-shell" data-compare-shell data-task-id="${task.id}" style="--split:${compareSplit}%">
              ${renderImage({
                className: "compare-image compare-before",
                src: activeSample.input,
                alt: `${task.title} input for ${activeSample.label}`,
                loading: "eager",
                fetchPriority: "high"
              })}
              ${renderImage({
                className: "compare-image compare-after",
                src: activeSample.output,
                alt: `${task.title} output for ${activeSample.label}`,
                loading: "eager",
                fetchPriority: "high"
              })}
              <div class="compare-label before-label">Input</div>
              <div class="compare-label after-label">RealRestorer</div>
              <div class="compare-divider"></div>
              <input class="compare-range" type="range" min="0" max="100" value="${compareSplit}" aria-label="Adjust before and after comparison">
            </div>

            <div class="reel-track" aria-label="${task.title} samples">
              ${task.samples
                .map((sample, sampleIndex) => {
                  const activeClass = sampleIndex === activeSampleIndex ? "reel-card is-active" : "reel-card";
                  return `
                    <button class="${activeClass}" type="button" data-sample-switch="${task.id}:${sampleIndex}" aria-pressed="${sampleIndex === activeSampleIndex}">
                      ${renderImage({
                        src: sample.output,
                        alt: `${task.title} ${sample.label}`,
                        loading: "lazy",
                        fetchPriority: "low",
                        variant: "thumb"
                      })}
                    </button>
                  `;
                })
                .join("")}
            </div>
          </section>

          <button class="showcase-side-btn showcase-side-btn-next" type="button" data-task-nav="next" aria-label="Show next case group">
            <span class="nav-btn-icon" aria-hidden="true">→</span>
          </button>
        </div>
      </div>
    `;

    bindTaskNav();
    bindSampleSwitch();
    bindCompareAll();
  }

  function renderResultsPlaceholder() {
    const resultsRoot = q("[data-results-root]");
    if (!resultsRoot || hasRenderedResults) {
      return;
    }

    resultsRoot.innerHTML = `
      <div class="results-placeholder" aria-hidden="true">
        <div class="results-placeholder-meta"></div>
        <div class="results-placeholder-card"></div>
      </div>
    `;
  }

  function initDeferredResults() {
    const resultsSection = q("#results");
    if (!resultsSection || showcaseData.length === 0) {
      return;
    }

    if (!("IntersectionObserver" in window)) {
      renderResults();
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;
        if (!entry || !entry.isIntersecting) {
          return;
        }

        renderResults();
        observer.disconnect();
      },
      {
        rootMargin: "320px 0px"
      }
    );

    observer.observe(resultsSection);
  }

  function bindTaskNav() {
    document.querySelectorAll("[data-task-nav]").forEach((button) => {
      button.addEventListener("click", () => {
        const direction = button.dataset.taskNav === "prev" ? "prev" : "next";
        lastTaskNavDirection = direction;
        activeTaskIndex = getWrappedTaskIndex(activeTaskIndex + (direction === "prev" ? -1 : 1));
        renderResults({ animateTaskChange: true });
      });
    });
  }

  function bindSampleSwitch() {
    document.querySelectorAll("[data-sample-switch]").forEach((button) => {
      button.addEventListener("click", () => {
        const [taskId, sampleIndex] = button.dataset.sampleSwitch.split(":");
        sampleState[taskId] = Number(sampleIndex);
        renderResults({ animateTaskChange: false });
      });
    });
  }

  function bindCompareAll() {
    document.querySelectorAll("[data-compare-shell]").forEach((compareShell) => {
      const compareRange = compareShell.querySelector(".compare-range");
      const taskId = compareShell.dataset.taskId;
      if (!compareRange) {
        return;
      }

      const sync = (value) => {
        compareState[taskId] = Number(value);
        compareShell.style.setProperty("--split", `${value}%`);
      };

      const updateFromPointer = (clientX) => {
        const rect = compareShell.getBoundingClientRect();
        const clamped = Math.max(0, Math.min(rect.width, clientX - rect.left));
        const percent = ((clamped / rect.width) * 100).toFixed(2);
        compareRange.value = percent;
        sync(percent);
      };

      sync(compareRange.value);

      compareRange.addEventListener("input", (event) => {
        sync(event.target.value);
      });

      compareShell.addEventListener("pointerdown", (event) => {
        updateFromPointer(event.clientX);

        const handleMove = (moveEvent) => {
          updateFromPointer(moveEvent.clientX);
        };

        const stopMove = () => {
          window.removeEventListener("pointermove", handleMove);
        };

        window.addEventListener("pointermove", handleMove);
        window.addEventListener("pointerup", stopMove, { once: true });
      });
    });
  }

  function init() {
    renderRanking(getSortedRanking());
    renderResultsPlaceholder();
    initDeferredResults();
  }

  window.addEventListener("DOMContentLoaded", init);
})();
