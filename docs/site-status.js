(() => {
  "use strict";

  const button = document.getElementById("download-button");
  const detail = document.getElementById("download-detail");
  if (!button || !detail) return;

  const sha256Pattern = /^[a-f0-9]{64}$/i;
  const blockedMessage = "Download remains disabled until every scientific, format, and distinctness gate is complete.";

  fetch("submission-status.json", { cache: "no-store" })
    .then((response) => {
      if (!response.ok) throw new Error(`status HTTP ${response.status}`);
      return response.json();
    })
    .then((status) => {
      const format = status.format_receipt;
      const holdout = status.holdout_receipt;
      const correlation = status.correlation_receipt;
      const interval = holdout && holdout.bootstrap_95_percent_interval;
      const priorReports = correlation && correlation.reports;
      const priorRowsPass = Array.isArray(priorReports)
        && Number.isInteger(correlation.prior_count)
        && correlation.prior_count > 0
        && priorReports.length === correlation.prior_count
        && priorReports.every((row) => row.correlation_gate_failed === false
          && Math.abs(row.pearson_all_pixels) < 0.90
          && Math.abs(row.spearman_systematic_sample) < 0.90);
      const holdoutPass = status.holdout_passed === true
        && holdout && holdout.promotion_gate_passed === true
        && holdout.positive_folds >= 3
        && holdout.mean_paired_delta > 0
        && Array.isArray(interval) && interval.length === 2 && interval[0] > 0;
      const gatesPass = status.ready === true
        && holdoutPass
        && typeof status.candidate_id === "string"
        && status.candidate_id.length > 0
        && typeof status.filename === "string"
        && status.filename.length > 0
        && typeof status.download_path === "string"
        && status.download_path.length > 0
        && typeof status.sha256 === "string"
        && sha256Pattern.test(status.sha256)
        && format && format.valid === true && format.sha256 === status.sha256
        && correlation && correlation.all_priors_distinct === true
        && priorRowsPass;

      if (!gatesPass) {
        detail.textContent = status.reason || blockedMessage;
        return;
      }

      const url = new URL(status.download_path, window.location.href);
      const isSafe = url.origin === window.location.origin || url.protocol === "https:";
      if (!isSafe) {
        detail.textContent = blockedMessage;
        return;
      }

      const filename = status.filename || url.pathname.split("/").filter(Boolean).pop();
      button.href = url.href;
      button.classList.remove("button-disabled");
      button.classList.add("button-light");
      button.removeAttribute("aria-disabled");
      button.removeAttribute("tabindex");
      button.setAttribute("download", filename || "GEMSDOE42-WPH01.tif");
      button.textContent = `Download ${filename || "validated GeoTIFF"}`;
      detail.textContent = `Candidate ${status.candidate_id} · SHA-256 ${status.sha256} · holdout and ${correlation.prior_count} prior-raster checks passed.`;
    })
    .catch(() => {
      detail.textContent = blockedMessage;
    });
})();
