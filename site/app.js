let data = null;
let activeRegion = null;
let activeForecastCadence = "weekly";
let activeForecastRegion = "us";
let explorerChartState = { points: [], emptyMessage: "" };
let explorerHitTargets = [];
let explorerHover = null;
let chartHitTargets = [];
let chartHover = null;
let latestSiteDataCsv = "";

const configuredSiteDataRefreshMs = Number(window.CONSUMERSIM_SITE_DATA_REFRESH_MS);
const SITE_DATA_REFRESH_MS = Number.isFinite(configuredSiteDataRefreshMs) ? configuredSiteDataRefreshMs : 60_000;

const els = {
  generatedAt: document.querySelector("#generated-at"),
  nextUpdate: document.querySelector("#next-update"),
  pageTitle: document.querySelector("#page-title"),
  typingStatus: document.querySelector("#typing-status"),
  mapValues: {
    us: document.querySelector("#map-us-value"),
    eu: document.querySelector("#map-eu-value"),
    jp: document.querySelector("#map-jp-value")
  },
  weeklyTitle: document.querySelector("#weekly-title"),
  weeklyContext: document.querySelector("#weekly-context"),
  weeklyChart: document.querySelector("#weekly-chart"),
  weeklyChartTooltip: document.querySelector("#weekly-chart-tooltip"),
  forecastNewsPeriod: document.querySelector("#forecast-news-period"),
  forecastNewsList: document.querySelector("#forecast-news-list"),
  weeklyNote: document.querySelector("#weekly-note"),
  cadenceTabs: document.querySelector("#cadence-tabs"),
  forecastRegionTabs: document.querySelector("#forecast-region-tabs"),
  regionTabs: document.querySelector("#region-tabs"),
  activeMarket: document.querySelector("#active-market"),
  chartTitle: document.querySelector("#chart-title"),
  chart: document.querySelector("#forecast-chart"),
  chartTooltip: document.querySelector("#chart-tooltip"),
  regionTitle: document.querySelector("#region-title"),
  regionHeadline: document.querySelector("#region-headline"),
  briefMae: document.querySelector("#brief-mae"),
  briefRmse: document.querySelector("#brief-rmse"),
  briefPearson: document.querySelector("#brief-pearson"),
  regionOutlineMap: document.querySelector("#region-outline-map"),
  tableContext: document.querySelector("#table-context"),
  historicalBody: document.querySelector("#historical-body"),
  agentPins: document.querySelector("#agent-pins"),
  agentProfilePanel: document.querySelector("#agent-profile-panel")
};

const regionOutlineViews = {
  us: "./assets/region-us.svg",
  eu: "./assets/region-eu.svg",
  jp: "./assets/region-jp.svg"
};

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"' && quoted && next === '"') {
      cell += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(cell);
      if (row.some((value) => value.trim() !== "")) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }

  row.push(cell);
  if (row.some((value) => value.trim() !== "")) rows.push(row);

  const headers = rows.shift().map((header) => header.trim());
  return rows.map((values) =>
    Object.fromEntries(headers.map((header, index) => [header, (values[index] ?? "").trim()]))
  );
}

function num(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function text(value) {
  return value || "";
}

function fmt(value, digits = 3) {
  const parsed = num(value);
  return parsed === null ? "TBD" : parsed.toFixed(digits);
}

function signed(value) {
  const parsed = num(value);
  if (parsed === null) return "TBD";
  const sign = parsed > 0 ? "+" : "";
  return `${sign}${fmt(parsed, 2)}`;
}

function bySort(a, b) {
  return (num(a.sort_order) ?? 0) - (num(b.sort_order) ?? 0);
}

function latestAsOf(rows) {
  const requested = new URLSearchParams(window.location.search).get("as_of");
  if (requested && rows.some((row) => row.as_of === requested)) return requested;
  return [...new Set(rows.map((row) => row.as_of))].sort().at(-1);
}

function buildData(rows) {
  const asOf = latestAsOf(rows);
  const current = rows.filter((row) => row.as_of === asOf);
  const metaRows = current.filter((row) => row.record_type === "meta");
  const meta = Object.fromEntries(metaRows.map((row) => [row.key, row.value]));

  const regions = current
    .filter((row) => row.record_type === "region_summary")
    .sort(bySort)
    .map((row) => {
      const region = row.region;
      const leaderboard = current
        .filter((item) => item.record_type === "leaderboard" && item.region === region)
        .sort(bySort)
        .map((item) => ({
          rank: num(item.rank),
          method: item.method || item.label,
          family: item.method ? item.family : item.period,
          months: num(item.months),
          mae: num(item.mae),
          rmse: num(item.rmse),
          pearson: num(item.pearson)
        }));
      const series = current
        .filter((item) => item.record_type === "series" && item.region === region)
        .sort(bySort)
        .map((item) => ({
          period: item.period || item.week_label,
          forecast: num(item.forecast) ?? num(item.actual),
          actual: item.forecast ? num(item.actual) : num(item.error)
        }));
      return {
        id: region,
        label: row.label,
        market: row.market,
        target: row.target,
        window: row.window,
        method: row.method,
        rankNote: row.signal,
        headline: row.interpretation,
        latest: {
          period: row.period,
          forecast: num(row.forecast),
          actual: num(row.actual),
          error: num(row.error)
        },
        stats: {
          months: num(row.months),
          mae: num(row.mae),
          rmse: num(row.rmse),
          pearson: num(row.pearson)
        },
        leaderboard,
        series
      };
    });

  const agentPredictions = current
    .filter((row) => row.record_type === "agent_monthly_prediction")
    .sort(bySort)
    .map((row) => ({
      id: row.key,
      period: row.period,
      value: num(row.value) ?? num(row.forecast),
      signal: row.signal
    }));

  const agentProfiles = current
    .filter((row) => row.record_type === "agent_profile")
    .sort(bySort)
    .map((row) => ({
      id: row.key,
      name: row.label,
      state: row.market,
      stateCode: row.period,
      location: row.week_label,
      age: num(row.cutoff_day),
      segment: row.target,
      education: row.method,
      income: row.family,
      household: row.prior_period,
      summary: row.signal,
      profile: row.interpretation,
      status: row.status,
      latest: num(row.value),
      x: num(row.forecast),
      y: num(row.actual),
      note: row.note,
      monthly: agentPredictions.filter((item) => item.id === row.key)
    }));

  return {
    generatedAt: meta.generatedAt,
    nextUpdate: meta.nextUpdate,
    monthlyPredictions: current
      .filter((row) => row.record_type === "monthly_prediction")
      .sort(bySort)
      .map((row) => ({
        id: row.region,
        label: row.label,
        period: row.period,
        valueLabel: row.value_label || fmt(row.value, 2),
        priorPeriod: row.prior_period,
        signal: row.signal,
        interpretation: row.interpretation
      })),
    weeklyPredictions: current
      .filter((row) => row.record_type === "weekly_prediction")
      .sort(bySort)
      .map((row) => ({
        id: row.region,
        label: row.week_label,
        period: row.period,
        cutoffDay: num(row.cutoff_day),
        forecast: num(row.forecast),
        currentConditions: num(row.actual),
        expectations: num(row.error),
        signal: row.signal,
        interpretation: row.interpretation
      })),
    forecastNews: current
      .filter((row) => row.record_type === "forecast_news")
      .sort(bySort)
      .map((row) => ({
        id: row.region,
        cadence: row.key,
        period: row.period || row.week_label,
        headline: row.label,
        source: row.market,
        tag: row.signal,
        summary: row.interpretation,
        url: row.note
      })),
    agentProfiles,
    regions
  };
}

function setMeta() {
  els.generatedAt.textContent = data.generatedAt;
  els.nextUpdate.textContent = data.nextUpdate;
}

function typeHeroTitle() {
  const node = els.pageTitle;
  if (!node) return;
  const fullText = node.dataset.title || node.textContent;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reducedMotion) {
    node.textContent = fullText;
    return;
  }

  node.textContent = "";
  node.classList.add("typing");
  let index = 0;
  const tick = () => {
    node.textContent = fullText.slice(0, index);
    index += 1;
    if (index <= fullText.length) {
      window.setTimeout(tick, index < 12 ? 38 : 26);
    } else {
      node.classList.remove("typing");
    }
  };
  tick();
}

function typeLoop(node, phrases) {
  if (!node || !phrases.length) return;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reducedMotion) {
    node.textContent = phrases[0];
    return;
  }

  let phraseIndex = 0;
  let charIndex = 0;
  let deleting = false;

  const tick = () => {
    const phrase = phrases[phraseIndex];
    node.textContent = phrase.slice(0, charIndex);

    if (!deleting && charIndex < phrase.length) {
      charIndex += 1;
      window.setTimeout(tick, 34);
      return;
    }

    if (!deleting && charIndex === phrase.length) {
      deleting = true;
      window.setTimeout(tick, 1500);
      return;
    }

    if (deleting && charIndex > 0) {
      charIndex -= 1;
      window.setTimeout(tick, 18);
      return;
    }

    deleting = false;
    phraseIndex = (phraseIndex + 1) % phrases.length;
    window.setTimeout(tick, 280);
  };

  tick();
}

function startTypingStatus() {
  typeLoop(els.typingStatus, [
    "reading the latest forecast snapshot",
    "plotting US, EU27, and Japan sentiment paths",
    "updating weekly nowcast signals"
  ]);
}

function renderMapValues() {
  data.monthlyPredictions.forEach((item) => {
    const node = els.mapValues[item.id];
    if (node) node.textContent = fmt(item.valueLabel, 2);
  });
}

function renderHeroForecasts() {
  const timeline = sharedHeroTimeline();
  data.regions.forEach((region) => {
    const monthly = data.monthlyPredictions.find((item) => item.id === region.id);
    const weekly = latestWeeklyForecast(region.id);
    const valueNode = document.querySelector(`[data-hero-value="${region.id}"]`);
    const periodNode = document.querySelector(`[data-hero-period="${region.id}"]`);
    const noteNode = document.querySelector(`[data-hero-note="${region.id}"]`);
    if (valueNode) valueNode.textContent = fmt(weekly?.forecast ?? region.latest.forecast, 2);
    if (periodNode) periodNode.textContent = weekly ? `${weekly.label} / ${weekly.period}` : monthly?.period || "--";
    if (noteNode) {
      noteNode.textContent = "";
      noteNode.hidden = true;
    }
    drawHeroForecastChart(region, timeline);
  });
}

function latestWeeklyForecast(regionId) {
  return latestWeeklyRows(regionId).at(-1);
}

function latestWeeklyRows(regionId) {
  const rows = data.weeklyPredictions.filter((point) => point.id === regionId);
  const latestPeriod = rows
    .map((point) => point.period)
    .filter(Boolean)
    .sort((a, b) => (monthDate(a)?.valueOf() || 0) - (monthDate(b)?.valueOf() || 0))
    .at(-1);
  return latestPeriod ? rows.filter((point) => point.period === latestPeriod) : rows;
}

function sharedHeroTimeline() {
  const actualPeriodSets = data.regions.map(
    (region) => new Set(region.series.filter((p) => Number.isFinite(p.actual)).map((p) => p.period))
  );
  const firstSeries = data.regions[0]?.series || [];
  const commonMonthly = firstSeries
    .map((p) => p.period)
    .filter((period) => actualPeriodSets.every((set) => set.has(period)));
  const monthlyPeriods = commonMonthly.slice(-10);
  const lastMonthlyDate = monthlyPeriods.length ? monthDate(monthlyPeriods.at(-1)) : null;

  const weeklyRowsAfter = (regionId) =>
    latestWeeklyRows(regionId).filter((p) => {
      const d = weeklyDate(p);
      return d && (!lastMonthlyDate || d.valueOf() > lastMonthlyDate.valueOf());
    });

  const weeklyMaps = data.regions.map((r) => new Map(weeklyRowsAfter(r.id).map((p) => [p.label, p])));
  let weeklyLabels = weeklyRowsAfter(data.regions[0]?.id || "")
    .map((p) => p.label)
    .filter((label) => weeklyMaps.every((map) => map.has(label)));

  if (!weeklyLabels.length) {
    const allMaps = data.regions.map((r) => new Map(latestWeeklyRows(r.id).map((p) => [p.label, p])));
    weeklyLabels = latestWeeklyRows(data.regions[0]?.id || "")
      .map((p) => p.label)
      .filter((label) => allMaps.every((map) => map.has(label)))
      .slice(-4);
  }
  return { monthlyPeriods, weeklyLabels };
}

function heroForecastSeries(region, timeline) {
  const seriesByPeriod = new Map(region.series.map((p) => [p.period, p]));
  const weeklyByLabel = new Map(latestWeeklyRows(region.id).map((p) => [p.label, p]));

  const monthly = timeline.monthlyPeriods
    .map((period) => {
      const p = seriesByPeriod.get(period);
      if (!p) return null;
      return { label: period, actual: p.actual, forecast: p.forecast };
    })
    .filter(Boolean);

  const weekly = timeline.weeklyLabels
    .map((label) => {
      const p = weeklyByLabel.get(label);
      if (!p) return null;
      return { label, forecast: p.forecast };
    })
    .filter(Boolean);

  return { monthly, weekly };
}

function monthDate(label) {
  const match = /^([A-Za-z]{3})-(\d{2})$/.exec(label || "");
  if (!match) return null;
  const month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].indexOf(match[1]);
  if (month < 0) return null;
  return new Date(2000 + Number(match[2]), month + 1, 0);
}

function weeklyDate(point) {
  const targetMatch = /^([A-Za-z]{3})-(\d{2})$/.exec(point.period || "");
  const weekMatch = /^([A-Za-z]{3}) W\d+$/.exec(point.label || "");
  if (!targetMatch || !weekMatch || !Number.isFinite(point.cutoffDay)) return null;
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const targetMonth = months.indexOf(targetMatch[1]);
  const weekMonth = months.indexOf(weekMatch[1]);
  if (targetMonth < 0 || weekMonth < 0) return null;
  let year = 2000 + Number(targetMatch[2]);
  if (targetMonth === 0 && weekMonth === 11) year -= 1;
  return new Date(year, weekMonth, point.cutoffDay);
}

function drawHeroForecastChart(region, timeline = sharedHeroTimeline()) {
  const canvas = document.querySelector(`[data-hero-chart="${region.id}"]`);
  if (!canvas) return;
  const { monthly, weekly } = heroForecastSeries(region, timeline);
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(280, Math.floor(rect.width));
  const height = Math.max(200, Math.floor(rect.height || 210));
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  if (monthly.length + weekly.length < 2) return;

  const values = [
    ...monthly.flatMap((p) => [p.actual, p.forecast]),
    ...weekly.map((p) => p.forecast)
  ].filter((value) => Number.isFinite(value));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = max - min || 1;
  const yMin = min - spread * 0.18;
  const yMax = max + spread * 0.18;
  const pad = { top: 14, right: 16, bottom: 32, left: 44 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  // Unified aligned dual-zone X geometry: identical across US, EU27, and Japan
  const hasWeekly = weekly.length > 0;
  const histEndFrac = hasWeekly ? 0.62 : 1.0;
  const dividerFrac = 0.665;
  const weekStartFrac = 0.715;
  const xForMonthly = (i) =>
    pad.left + (monthly.length <= 1 ? 0 : (i / (monthly.length - 1)) * (plotW * histEndFrac));
  const xForWeekly = (j) =>
    pad.left +
    plotW * weekStartFrac +
    (weekly.length <= 1 ? (plotW * (1 - weekStartFrac)) / 2 : (j / (weekly.length - 1)) * (plotW * (1 - weekStartFrac)));
  const yFor = (value) => pad.top + ((yMax - value) / (yMax - yMin)) * plotH;

  // Subtle tinted backdrop for the aligned live weekly forecast zone
  if (hasWeekly) {
    const divX = pad.left + plotW * dividerFrac;
    ctx.save();
    ctx.fillStyle = "rgba(214, 79, 42, 0.045)";
    ctx.fillRect(divX, pad.top, width - pad.right - divX, plotH);
    ctx.restore();
  }

  ctx.font = "11px Consolas, monospace";
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#dedbd2";
  ctx.fillStyle = "#6b6860";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const yDigits = yMax - yMin < 4 || region.id === "eu" ? 1 : 0;
  for (let i = 0; i < 3; i += 1) {
    const t = i / 2;
    const y = pad.top + t * plotH;
    const value = yMax - t * (yMax - yMin);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillText(value.toFixed(yDigits), pad.left - 8, y);
  }

  if (hasWeekly) {
    const divX = pad.left + plotW * dividerFrac;
    ctx.save();
    ctx.setLineDash([4, 5]);
    ctx.strokeStyle = "rgba(16, 16, 16, 0.24)";
    ctx.beginPath();
    ctx.moveTo(divX, pad.top);
    ctx.lineTo(divX, pad.top + plotH);
    ctx.stroke();
    ctx.restore();
  }

  const actualPoints = monthly
    .map((p, i) => (Number.isFinite(p.actual) ? [xForMonthly(i), yFor(p.actual)] : null))
    .filter(Boolean);
  const monthlyForecastPoints = monthly
    .map((p, i) => (Number.isFinite(p.forecast) ? [xForMonthly(i), yFor(p.forecast)] : null))
    .filter(Boolean);
  const weeklyForecastPoints = weekly
    .map((p, j) => (Number.isFinite(p.forecast) ? [xForWeekly(j), yFor(p.forecast)] : null))
    .filter(Boolean);

  drawLine(ctx, actualPoints, "#101010", 2.0);
  drawLine(ctx, monthlyForecastPoints, "#d64f2a", 1.9);
  drawDashedLine(ctx, weeklyForecastPoints, "#d64f2a", 2.2);
  drawPoints(ctx, actualPoints, "#101010", 2.5);
  drawPoints(ctx, monthlyForecastPoints, "#d64f2a", 2.4);
  drawPoints(ctx, weeklyForecastPoints, "#d64f2a", 3.0);

  // Aligned X-axis tick labels (identical slots across all 3 cards)
  ctx.fillStyle = "#6b6860";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const ticks = [];
  if (monthly.length) {
    ticks.push({ label: monthly[0].label, x: xForMonthly(0) });
    if (monthly.length >= 5) {
      const midIdx = Math.floor((monthly.length - 1) / 2);
      ticks.push({ label: monthly[midIdx].label, x: xForMonthly(midIdx) });
    }
    ticks.push({ label: monthly.at(-1).label, x: xForMonthly(monthly.length - 1) });
  }
  if (weekly.length) {
    ticks.push({ label: weekly.at(-1).label, x: xForWeekly(weekly.length - 1) });
  }
  let lastLabelX = -Infinity;
  ticks.forEach((tick) => {
    const lx = Math.max(pad.left + 18, Math.min(width - pad.right - 18, tick.x));
    if (Math.abs(lx - lastLabelX) >= 48) {
      ctx.fillText(tick.label, lx, pad.top + plotH + 10);
      lastLabelX = lx;
    }
  });
}

function renderAgentDemo() {
  if (!els.agentPins || !els.agentProfilePanel || !data.agentProfiles.length) return;
  els.agentPins.innerHTML = data.agentProfiles
    .map((agent, index) => {
      const left = ((agent.x || 0) / 760) * 100;
      const top = ((agent.y || 0) / 430) * 100;
      return `
        <button class="agent-pin ${index === 0 ? "active" : ""}" type="button" data-agent-id="${agent.id}"
          style="--x: ${left.toFixed(2)}%; --y: ${top.toFixed(2)}%;" aria-label="${agent.name}, ${agent.state}">
          <span class="agent-dot">${initials(agent.name)}</span>
          <span class="agent-state">${agent.stateCode}</span>
          <span class="agent-hover-card">
            <strong>${agent.name}</strong>
            <span>${agent.state} · age ${agent.age}</span>
            <em>${agent.summary}</em>
          </span>
        </button>
      `;
    })
    .join("");

  els.agentPins.querySelectorAll(".agent-pin").forEach((button) => {
    const setActive = () => setActiveAgent(button.dataset.agentId);
    button.addEventListener("mouseenter", setActive);
    button.addEventListener("focus", setActive);
    button.addEventListener("click", () => {
      pulseButton(button);
      setActive();
    });
  });

  renderAgentPanel(data.agentProfiles[0]);
}

function setActiveAgent(agentId) {
  const agent = data.agentProfiles.find((item) => item.id === agentId);
  if (!agent) return;
  els.agentPins.querySelectorAll(".agent-pin").forEach((button) => {
    button.classList.toggle("active", button.dataset.agentId === agentId);
  });
  renderAgentPanel(agent);
}

function renderAgentPanel(agent) {
  const values = agent.monthly.map((item) => item.value).filter((value) => Number.isFinite(value));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = max - min || 1;
  els.agentProfilePanel.innerHTML = `
    <div class="agent-panel-top">
      <p class="eyebrow">${agent.state} sample</p>
      <h3>${agent.name}</h3>
      <strong>${fmt(agent.latest, 1)}</strong>
    </div>
    <p class="agent-summary">${agent.summary}</p>
    <dl class="agent-profile-grid">
      <div><dt>Age</dt><dd>${agent.age}</dd></div>
      <div><dt>Location</dt><dd>${agent.location}</dd></div>
      <div><dt>Status</dt><dd>${agent.status}</dd></div>
      <div><dt>Education</dt><dd>${agent.education}</dd></div>
      <div><dt>Income</dt><dd>${agent.income}</dd></div>
      <div><dt>Household</dt><dd>${agent.household}</dd></div>
    </dl>
    <p class="agent-profile-copy">${agent.profile}</p>
    <div class="agent-monthly-list" aria-label="${agent.name} monthly predicted survey responses">
      ${agent.monthly
        .map((point) => {
          const width = 18 + ((point.value - min) / spread) * 82;
          return `
            <div class="agent-month">
              <span>${point.period}</span>
              <i><b style="width: ${width.toFixed(1)}%;"></b></i>
              <strong>${fmt(point.value, 1)}</strong>
            </div>
          `;
        })
        .join("")}
    </div>
    <small>${agent.note}</small>
  `;
}

function initials(name) {
  return text(name)
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0])
    .join("")
    .toUpperCase();
}

function setupReveal() {
  const targets = document.querySelectorAll(
    ".prediction-section, .section-head, .dashboard-grid, .table-section, .updates, .map-card, .agent-section"
  );
  targets.forEach((target) => target.classList.add("reveal"));

  if (!("IntersectionObserver" in window)) {
    targets.forEach((target) => target.classList.add("is-visible"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.05 }
  );
  targets.forEach((target) => {
    const rect = target.getBoundingClientRect();
    if (rect.top < window.innerHeight * 1.05) {
      target.classList.add("is-visible");
    } else {
      observer.observe(target);
    }
  });
}

function createTabs() {
  els.regionTabs.innerHTML = data.regions
    .map(
      (region) => `
        <button type="button" role="tab" data-region="${region.id}" aria-selected="false">
          ${region.label}
        </button>
      `
    )
    .join("");

  els.regionTabs.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      pulseButton(button);
      setActiveRegion(button.dataset.region);
    });
  });
}

function createForecastControls() {
  els.cadenceTabs.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      pulseButton(button);
      clearExplorerTooltip(false);
      activeForecastCadence = button.dataset.cadence;
      renderForecastExplorer();
    });
  });

  els.forecastRegionTabs.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      pulseButton(button);
      clearExplorerTooltip(false);
      activeForecastRegion = button.dataset.forecastRegion;
      renderForecastExplorer();
    });
  });
}

function pulseButton(button) {
  button.classList.remove("pressed");
  void button.offsetWidth;
  button.classList.add("pressed");
  window.setTimeout(() => button.classList.remove("pressed"), 430);
}

function updateForecastControls() {
  els.cadenceTabs.querySelectorAll("button").forEach((button) => {
    button.setAttribute("aria-selected", String(button.dataset.cadence === activeForecastCadence));
  });

  els.forecastRegionTabs.querySelectorAll("button").forEach((button) => {
    button.setAttribute("aria-selected", String(button.dataset.forecastRegion === activeForecastRegion));
  });
}

function regionForExplorer() {
  return data.regions.find((region) => region.id === activeForecastRegion) || data.regions[0];
}

function monthlyForExplorer() {
  return data.monthlyPredictions.find((item) => item.id === activeForecastRegion);
}

function newsForExplorer(region, cadence) {
  return data.forecastNews.filter((item) => item.id === region.id && item.cadence === cadence && item.headline);
}

function renderForecastNews(region, cadence, fallbackPeriod) {
  const items = newsForExplorer(region, cadence);
  const period = items[0]?.period || fallbackPeriod || "--";
  els.forecastNewsPeriod.textContent = period;

  if (!items.length) {
    els.forecastNewsList.innerHTML = `
      <article class="news-item empty-news">
        <span>No latest driver notes</span>
        <h3>Drivers have not been added for this view.</h3>
        <p>Driver notes will appear after the next published update.</p>
      </article>
    `;
    return;
  }

  els.forecastNewsList.innerHTML = items
    .map(
      (item) => `
        <article class="news-item">
          <div class="news-item-top">
            <span>${item.tag || "Driver"}</span>
            ${
              item.url
                ? `<a href="${item.url}" target="_blank" rel="noreferrer">${item.source || "Source"}</a>`
                : `<small>${item.source || "Source"}</small>`
            }
          </div>
          <h3>${item.headline}</h3>
          <p>${shortNewsSummary(item.summary)}</p>
        </article>
      `
    )
    .join("");
}

function shortNewsSummary(summary) {
  const textValue = text(summary);
  if (textValue.length <= 118) return textValue;
  const clipped = textValue.slice(0, 116);
  return `${clipped.slice(0, clipped.lastIndexOf(" "))}...`;
}

function renderUnavailableExplorer(region) {
  const cadenceLabel = activeForecastCadence === "monthly" ? "Monthly" : "Weekly";
  els.weeklyTitle.textContent = `${region.label} ${activeForecastCadence} forecast`;
  els.weeklyContext.textContent = `${cadenceLabel} view is not configured yet for ${region.label}.`;
  els.forecastNewsPeriod.textContent = "--";
  els.forecastNewsList.innerHTML = `
    <article class="news-item empty-news">
      <span>${cadenceLabel}</span>
      <h3>News panel is not configured for this view.</h3>
      <p>EU27 and Japan monthly views can be added later through the site data file.</p>
    </article>
  `;
  els.weeklyNote.textContent =
    activeForecastCadence === "monthly"
      ? `${region.label} monthly rows will appear here once the regional monthly forecast records are added.`
      : "Weekly rows will appear here once the regional weekly forecast records are added.";
  explorerChartState = {
    points: [],
    emptyMessage: `${cadenceLabel} view not configured`
  };
  drawExplorerChart();
}

function renderWeeklyExplorer(region) {
  const rows = latestWeeklyRows(activeForecastRegion);
  if (!rows.length) {
    renderUnavailableExplorer(region);
    return;
  }

  const firstW = rows[0].label;
  const lastW = rows.at(-1).label;
  const targetP = rows.at(-1).period;
  els.weeklyTitle.textContent = `${region.label} weekly consumer sentiment nowcast`;
  els.weeklyContext.textContent = `${region.label}, ${firstW} to ${lastW} (target ${targetP}). Intra-month weekly nowcast slots.`;
  renderForecastNews(region, "weekly", lastW);
  els.weeklyNote.textContent =
    rows.at(-1)?.interpretation ||
    `${region.label} weekly nowcast trajectory through ${lastW}.`;
  explorerChartState = {
    points: rows.map((row) => ({ label: row.label, value: row.forecast })),
    emptyMessage: ""
  };
  drawExplorerChart();
}

function renderMonthlyExplorer(region) {
  const monthly = monthlyForExplorer();
  const rows = region.series.slice(-6);
  if (!monthly || !rows.length) {
    renderUnavailableExplorer(region);
    return;
  }

  els.weeklyTitle.textContent = `${region.label} monthly consumer sentiment path`;
  els.weeklyContext.textContent = `Latest monthly forecast: ${monthly.period} at ${fmt(monthly.valueLabel, 2)}.`;
  renderForecastNews(region, "monthly", monthly.period);
  els.weeklyNote.textContent = monthly.interpretation;
  explorerChartState = {
    points: rows.map((row) => ({ label: row.period, value: row.forecast })),
    emptyMessage: ""
  };
  drawExplorerChart();
}

function renderForecastExplorer() {
  updateForecastControls();
  const region = regionForExplorer();
  if (activeForecastCadence === "weekly") {
    renderWeeklyExplorer(region);
  } else {
    renderMonthlyExplorer(region);
  }
}

function setActiveRegion(regionId) {
  const next = data.regions.find((region) => region.id === regionId);
  if (!next) return;
  clearChartTooltip();
  activeRegion = next;
  render();
}

function renderCardsAndTabs() {
  document.querySelectorAll("[data-region]").forEach((node) => {
    const isActive = node.dataset.region === activeRegion.id;
    node.classList.toggle("active", isActive);
    if (node.getAttribute("role") === "tab") {
      node.setAttribute("aria-selected", String(isActive));
    }
  });
}

function renderBrief() {
  els.activeMarket.textContent = `${activeRegion.label} / ${activeRegion.market}`;
  els.chartTitle.textContent = `${activeRegion.label}: forecast vs actual`;
  els.regionTitle.textContent = `${activeRegion.label} (${activeRegion.market})`;
  const firstMonth = activeRegion.series[0]?.period;
  const lastMonth = activeRegion.series.at(-1)?.period;
  els.regionHeadline.textContent =
    firstMonth && lastMonth ? `Displayed months: ${firstMonth} - ${lastMonth}.` : activeRegion.window;
  els.briefMae.textContent = fmt(activeRegion.stats.mae);
  els.briefRmse.textContent = fmt(activeRegion.stats.rmse);
  els.briefPearson.textContent = fmt(activeRegion.stats.pearson);
  renderRegionOutline();
}

function renderRegionOutline() {
  els.regionOutlineMap.src = regionOutlineViews[activeRegion.id] || regionOutlineViews.us;
}

function renderTable() {
  els.tableContext.textContent = `${shortTarget(activeRegion.target)} | ${activeRegion.window} | ${shortRankNote(
    activeRegion.rankNote
  )}.`;
  els.historicalBody.innerHTML = activeRegion.leaderboard
    .map((row, index) => {
      const isConsumerSim = row.method.toLowerCase().includes("consumersim");
      return `
        <tr class="${isConsumerSim ? "highlight-row" : ""} ${index >= 5 ? "extra-method" : ""}">
          <td class="num">${row.rank}</td>
          <td>${row.method}</td>
          <td><span class="family-pill">${row.family}</span></td>
          <td class="num">${row.months}</td>
          <td class="num">${fmt(row.mae)}</td>
          <td class="num">${fmt(row.rmse)}</td>
          <td class="num">${fmt(row.pearson)}</td>
        </tr>
      `;
    })
    .join("");
}

function shortTarget(target) {
  return target
    .replace("University of Michigan Index of Consumer Sentiment", "UMich ICS")
    .replace("Eurostat consumer confidence indicator", "Eurostat CCI")
    .replace("Cabinet Office / ESRI Consumer Confidence Index", "ESRI CCI");
}

function shortRankNote(note) {
  return note
    .replace("#1 by MAE among displayed non-naive methods", "#1 MAE among displayed non-naive methods")
    .replace("Leads the displayed model comparison for EU27.", "#1 MAE among displayed methods")
    .replace("Leads the displayed model comparison for Japan.", "#1 MAE among displayed methods");
}

function chartValues(series) {
  return series.flatMap((point) => [point.forecast, point.actual]).filter((value) => Number.isFinite(value));
}

function drawChart() {
  const canvas = els.chart;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(rect.width));
  const height = Math.max(320, Math.floor(rect.height || 340));
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const padding = { top: 28, right: 24, bottom: 54, left: 54 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;
  const values = chartValues(activeRegion.series);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = max - min || 1;
  const yMin = min - spread * 0.14;
  const yMax = max + spread * 0.14;

  const xFor = (index) =>
    padding.left + (activeRegion.series.length === 1 ? 0 : (index / (activeRegion.series.length - 1)) * plotW);
  const yFor = (value) => padding.top + ((yMax - value) / (yMax - yMin)) * plotH;
  chartHitTargets = [];

  ctx.font = "12px Consolas, monospace";
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#dedbd2";
  ctx.fillStyle = "#5f5f5f";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";

  for (let i = 0; i < 5; i += 1) {
    const t = i / 4;
    const value = yMax - t * (yMax - yMin);
    const y = padding.top + t * plotH;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
    ctx.fillText(value.toFixed(1), padding.left - 9, y);
  }

  ctx.strokeStyle = "#101010";
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, padding.top + plotH);
  ctx.lineTo(padding.left + plotW, padding.top + plotH);
  ctx.stroke();

  const forecastPoints = [];
  const actualPoints = [];
  activeRegion.series.forEach((point, index) => {
    const x = xFor(index);
    if (Number.isFinite(point.actual)) {
      const y = yFor(point.actual);
      actualPoints.push([x, y]);
      chartHitTargets.push({ type: "actual", index, x, y });
    }
    if (Number.isFinite(point.forecast)) {
      const y = yFor(point.forecast);
      forecastPoints.push([x, y]);
      chartHitTargets.push({ type: "forecast", index, x, y });
    }
  });

  drawLine(ctx, actualPoints, "#101010", 2.5);
  drawLine(ctx, forecastPoints, "#d64f2a", 2.5);
  drawPoints(ctx, actualPoints, "#101010");
  drawPoints(ctx, forecastPoints, "#d64f2a");
  drawHoverPoint(ctx);

  ctx.fillStyle = "#5f5f5f";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const maxTicks = Math.max(4, Math.floor(plotW / 68));
  const step = Math.max(1, Math.ceil(activeRegion.series.length / maxTicks));
  let prevTickX = -Infinity;
  activeRegion.series.forEach((point, i) => {
    const isLast = i === activeRegion.series.length - 1;
    if (i % step === 0 || isLast) {
      const tx = xFor(i);
      if (!isLast && tx - prevTickX < 46) return;
      if (isLast && tx - prevTickX < 38) return;
      const lbl = /^\d{4}-\d{2}$/.test(point.period) ? point.period.slice(2) : point.period;
      ctx.fillText(lbl, tx, padding.top + plotH + 16);
      prevTickX = tx;
    }
  });
}

function drawHoverPoint(ctx) {
  if (!chartHover) return;
  const target = chartHitTargets.find((item) => item.type === chartHover.type && item.index === chartHover.index);
  if (!target) return;
  const color = target.type === "forecast" ? "#d64f2a" : "#101010";
  ctx.save();
  ctx.beginPath();
  ctx.arc(target.x, target.y, 7.5, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255, 253, 248, 0.94)";
  ctx.fill();
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = color;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(target.x, target.y, 3.8, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.restore();
}

function setupChartTooltip() {
  els.chart.addEventListener("pointermove", handleChartPointerMove);
  els.chart.addEventListener("pointerleave", clearChartTooltip);
}

function handleChartPointerMove(event) {
  if (!activeRegion || !chartHitTargets.length) return;
  const rect = els.chart.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const nearest = nearestChartTarget(x, y);

  if (!nearest || nearest.distance > 16) {
    clearChartTooltip();
    return;
  }

  if (!chartHover || chartHover.type !== nearest.target.type || chartHover.index !== nearest.target.index) {
    chartHover = { type: nearest.target.type, index: nearest.target.index };
    drawChart();
  }
  showChartTooltip(nearest.target);
}

function nearestChartTarget(x, y) {
  return chartHitTargets.reduce((best, target) => {
    const distance = Math.hypot(target.x - x, target.y - y);
    return !best || distance < best.distance ? { target, distance } : best;
  }, null);
}

function showChartTooltip(target) {
  const point = activeRegion.series[target.index];
  const forecastClass = target.type === "forecast" ? "active" : "";
  const actualClass = target.type === "actual" ? "active" : "";
  els.chartTooltip.innerHTML = `
    <div class="tooltip-period">${point.period}</div>
    <div class="tooltip-row ${forecastClass}">
      <span>Forecast</span>
      <b>${valueLabel(point.forecast)}</b>
    </div>
    <div class="tooltip-row ${actualClass}">
      <span>Actual</span>
      <b>${valueLabel(point.actual)}</b>
    </div>
  `;
  els.chartTooltip.hidden = false;

  const canvasBox = els.chart.getBoundingClientRect();
  const shellBox = els.chart.parentElement.getBoundingClientRect();
  const tooltipWidth = els.chartTooltip.offsetWidth || 160;
  const tooltipHeight = els.chartTooltip.offsetHeight || 94;
  const rawLeft = canvasBox.left - shellBox.left + target.x + 14;
  const rawTop = canvasBox.top - shellBox.top + target.y - tooltipHeight - 12;
  const maxLeft = shellBox.width - tooltipWidth - 12;
  const maxTop = shellBox.height - tooltipHeight - 12;
  els.chartTooltip.style.left = `${Math.max(12, Math.min(rawLeft, maxLeft))}px`;
  els.chartTooltip.style.top = `${Math.max(12, Math.min(rawTop, maxTop))}px`;
}

function clearChartTooltip() {
  if (!chartHover && els.chartTooltip.hidden) return;
  chartHover = null;
  els.chartTooltip.hidden = true;
  if (activeRegion) drawChart();
}

function valueLabel(value) {
  return Number.isFinite(value) ? value.toFixed(2) : "pending";
}

function drawExplorerChart() {
  const canvas = els.weeklyChart;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(280, Math.floor(rect.width));
  const height = Math.max(240, Math.floor(rect.height || 260));
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const rows = explorerChartState.points.filter((row) => Number.isFinite(row.value));
  explorerHitTargets = [];
  if (rows.length < 2) {
    clearExplorerTooltip(false);
    ctx.fillStyle = "#5f5f5f";
    ctx.font = "13px Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(explorerChartState.emptyMessage || "No chart data", width / 2, height / 2);
    return;
  }

  const values = rows.map((row) => row.value);
  const min = Math.min(...values) - 1;
  const max = Math.max(...values) + 1;
  const pad = { top: 22, right: 30, bottom: 42, left: 48 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const xFor = (index) => pad.left + (index / (rows.length - 1)) * plotW;
  const yFor = (value) => pad.top + ((max - value) / (max - min)) * plotH;

  ctx.font = "12px Consolas, monospace";
  ctx.strokeStyle = "#dedbd2";
  ctx.fillStyle = "#5f5f5f";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i < 4; i += 1) {
    const t = i / 3;
    const y = pad.top + t * plotH;
    const value = max - t * (max - min);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillText(value.toFixed(1), pad.left - 8, y);
  }

  const forecastPoints = rows.map((row, index) => [xFor(index), yFor(row.value)]);
  explorerHitTargets = rows.map((row, index) => ({
    index,
    label: row.label,
    value: row.value,
    x: xFor(index),
    y: yFor(row.value)
  }));
  drawLine(ctx, forecastPoints, "#d64f2a", 2.8);
  drawPoints(ctx, forecastPoints, "#d64f2a");
  drawExplorerHoverPoint(ctx);

  ctx.fillStyle = "#5f5f5f";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  let prevExpX = -Infinity;
  rows.forEach((row, index) => {
    const x = xFor(index);
    const isLast = index === rows.length - 1;
    if (index === 0 || isLast || x - prevExpX >= 56) {
      if (isLast && x - prevExpX < 44) return;
      ctx.fillText(row.label, x, pad.top + plotH + 14);
      prevExpX = x;
    }
  });
}

function drawExplorerHoverPoint(ctx) {
  if (!explorerHover) return;
  const target = explorerHitTargets.find((item) => item.index === explorerHover.index);
  if (!target) return;
  ctx.save();
  ctx.beginPath();
  ctx.arc(target.x, target.y, 7.5, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255, 253, 248, 0.94)";
  ctx.fill();
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = "#d64f2a";
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(target.x, target.y, 3.8, 0, Math.PI * 2);
  ctx.fillStyle = "#d64f2a";
  ctx.fill();
  ctx.restore();
}

function setupExplorerTooltip() {
  els.weeklyChart.addEventListener("pointermove", handleExplorerPointerMove);
  els.weeklyChart.addEventListener("pointerleave", () => clearExplorerTooltip());
}

function handleExplorerPointerMove(event) {
  if (!explorerHitTargets.length) return;
  const rect = els.weeklyChart.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const nearest = explorerHitTargets.reduce((best, target) => {
    const distance = Math.hypot(target.x - x, target.y - y);
    return !best || distance < best.distance ? { target, distance } : best;
  }, null);

  if (!nearest || nearest.distance > 16) {
    clearExplorerTooltip();
    return;
  }

  if (!explorerHover || explorerHover.index !== nearest.target.index) {
    explorerHover = { index: nearest.target.index };
    drawExplorerChart();
  }
  showExplorerTooltip(nearest.target);
}

function showExplorerTooltip(target) {
  els.weeklyChartTooltip.innerHTML = `
    <div class="tooltip-period">${target.label}</div>
    <div class="tooltip-row active">
      <span>Forecast</span>
      <b>${valueLabel(target.value)}</b>
    </div>
  `;
  els.weeklyChartTooltip.hidden = false;

  const canvasBox = els.weeklyChart.getBoundingClientRect();
  const shellBox = els.weeklyChart.parentElement.getBoundingClientRect();
  const tooltipWidth = els.weeklyChartTooltip.offsetWidth || 142;
  const tooltipHeight = els.weeklyChartTooltip.offsetHeight || 62;
  const rawLeft = canvasBox.left - shellBox.left + target.x + 12;
  const rawTop = canvasBox.top - shellBox.top + target.y - tooltipHeight - 10;
  const maxLeft = shellBox.width - tooltipWidth - 10;
  const maxTop = shellBox.height - tooltipHeight - 10;
  els.weeklyChartTooltip.style.left = `${Math.max(10, Math.min(rawLeft, maxLeft))}px`;
  els.weeklyChartTooltip.style.top = `${Math.max(10, Math.min(rawTop, maxTop))}px`;
}

function clearExplorerTooltip(redraw = true) {
  if (!els.weeklyChartTooltip) return;
  const hadHover = !!explorerHover || !els.weeklyChartTooltip.hidden;
  explorerHover = null;
  els.weeklyChartTooltip.hidden = true;
  if (redraw && hadHover) drawExplorerChart();
}

function drawLine(ctx, points, color, width) {
  if (!points.length) return;
  ctx.beginPath();
  points.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.stroke();
}

function drawDashedLine(ctx, points, color, width) {
  if (!points.length) return;
  ctx.save();
  ctx.setLineDash([8, 6]);
  drawLine(ctx, points, color, width);
  ctx.restore();
}

function drawPoints(ctx, points, color, radius = 3.5) {
  ctx.fillStyle = color;
  points.forEach(([x, y]) => {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  });
}

function render() {
  renderCardsAndTabs();
  renderBrief();
  renderTable();
  drawChart();
}

function applySiteDataCsv(csv, initial = false) {
  if (!initial && csv === latestSiteDataCsv) return false;
  latestSiteDataCsv = csv;
  const previousRegionId = activeRegion?.id;
  data = buildData(parseCsv(csv));
  activeRegion = data.regions.find((region) => region.id === previousRegionId) || data.regions[0];
  if (!data.regions.some((region) => region.id === activeForecastRegion)) {
    activeForecastRegion = data.regions[0]?.id || activeForecastRegion;
  }

  if (!initial) {
    createTabs();
    setMeta();
    renderMapValues();
    renderHeroForecasts();
    renderAgentDemo();
    renderForecastExplorer();
    render();
  }
  return true;
}

function startSiteDataRefresh() {
  if (!Number.isFinite(SITE_DATA_REFRESH_MS) || SITE_DATA_REFRESH_MS <= 0) return;
  window.setInterval(async () => {
    try {
      applySiteDataCsv(await loadSiteDataCsv());
    } catch (error) {
      console.warn("Unable to refresh ConsumerSim site data.", error);
    }
  }, SITE_DATA_REFRESH_MS);
}

async function init() {
  applySiteDataCsv(await loadSiteDataCsv(), true);
  typeHeroTitle();
  startTypingStatus();
  setupReveal();
  setMeta();
  renderMapValues();
  renderHeroForecasts();
  createTabs();
  createForecastControls();
  setupChartTooltip();
  setupExplorerTooltip();
  renderAgentDemo();
  renderForecastExplorer();
  render();
  startSiteDataRefresh();
}

async function loadSiteDataCsv() {
  if (typeof window.CONSUMERSIM_INLINE_CSV === "string" && window.CONSUMERSIM_INLINE_CSV.trim()) {
    return window.CONSUMERSIM_INLINE_CSV;
  }
  const configuredSource = window.CONSUMERSIM_SITE_DATA_URL;
  const sources = [configuredSource, "./data/consumersim_site_data.csv", "/api/site-data"].filter(Boolean);
  let lastError = null;
  for (const source of sources) {
    try {
      const response = await fetch(source, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`${source} returned ${response.status}`);
      }
      return await response.text();
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error("Unable to load ConsumerSim site data.");
}

window.addEventListener("resize", () => {
  if (!data) return;
  window.requestAnimationFrame(() => {
    drawChart();
    drawExplorerChart();
    renderHeroForecasts();
  });
});

init();
