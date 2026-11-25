(() => {
  if (location.pathname !== "/portfolio-explorer") return;

  const portfolioOrigin = "https://musteermain.vercel.app";
  const root = document.documentElement;
  root.dataset.portfolioEmbed = "";

  const framed = window.parent !== window;
  if (framed) root.dataset.portfolioFramed = "";

  document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("page-title").textContent = "MRI scan explorer";
    if (!framed) return;

    const main = document.querySelector("main");
    let reportedHeight = 0;

    function sendHeight(force = false) {
      const height = Math.ceil(main.getBoundingClientRect().height);
      if (height > 0 && (height !== reportedHeight || force)) {
        reportedHeight = height;
        window.parent.postMessage({ type: "mri-explorer:size", height }, portfolioOrigin);
      }
    }

    new ResizeObserver(() => sendHeight()).observe(main);
    window.addEventListener("message", (event) => {
      if (event.origin === portfolioOrigin && event.source === window.parent &&
          event.data?.type === "mri-explorer:measure") {
        sendHeight(true);
      }
    });

    document.addEventListener("click", (event) => {
      const link = event.target.closest('.skip-link, #locate-scan, #open-model-input');
      if (!link || event.defaultPrevented || event.button !== 0 ||
          event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;

      const destination = document.getElementById(link.hash.slice(1));
      if (!destination || destination.hidden) return;
      event.preventDefault();

      if (link.id === "open-model-input") {
        destination.querySelector("summary").focus({ preventScroll: true });
      } else if (link.classList.contains("skip-link")) {
        document.getElementById("pick-image").focus({ preventScroll: true });
      }

      const top = destination.getBoundingClientRect().top + window.scrollY;
      window.parent.postMessage({ type: "mri-explorer:scroll", top }, portfolioOrigin);
    });

    sendHeight();
  });
})();
