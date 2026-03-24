// buildbanner.js — vendored from buildbanner v0.1.0
// Source: https://github.com/diziet/buildbanner
(() => {
  var __defProp = Object.defineProperty;
  var __defProps = Object.defineProperties;
  var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
  var __getOwnPropSymbols = Object.getOwnPropertySymbols;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __propIsEnum = Object.prototype.propertyIsEnumerable;
  var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
  var __spreadValues = (a, b) => {
    for (var prop in b || (b = {}))
      if (__hasOwnProp.call(b, prop))
        __defNormalProp(a, prop, b[prop]);
    if (__getOwnPropSymbols)
      for (var prop of __getOwnPropSymbols(b)) {
        if (__propIsEnum.call(b, prop))
          __defNormalProp(a, prop, b[prop]);
      }
    return a;
  };
  var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b));

  // src/config.js
  var VALID_POSITIONS = ["top", "bottom"];
  var VALID_THEMES = ["dark", "light", "auto"];
  var VALID_DISMISS = ["session", "permanent", "none"];
  var MIN_HEIGHT = 24;
  var MAX_HEIGHT = 48;
  var DEFAULT_CONFIG = Object.freeze({
    endpoint: "/buildbanner.json",
    position: "top",
    theme: "dark",
    dismiss: "session",
    envHide: null,
    height: 28,
    debug: false,
    poll: 0,
    push: true,
    token: null,
    manual: false,
    zIndex: 999999,
    hostPatterns: []
  });
  function _parseBool(value, defaultValue) {
    if (value == null) return defaultValue;
    const lower = String(value).toLowerCase().trim();
    if (lower === "false" || lower === "0" || lower === "no") return false;
    if (lower === "true" || lower === "1" || lower === "yes" || lower === "") return true;
    return defaultValue;
  }
  function _parseIntOrDefault(value, defaultValue) {
    if (value == null) return defaultValue;
    const parsed = parseInt(value, 10);
    if (Number.isNaN(parsed)) return defaultValue;
    return parsed;
  }
  function _parseHeight(value) {
    const parsed = _parseIntOrDefault(value, DEFAULT_CONFIG.height);
    return Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT, parsed));
  }
  function _parsePoll(value) {
    const parsed = _parseIntOrDefault(value, DEFAULT_CONFIG.poll);
    return parsed < 0 ? DEFAULT_CONFIG.poll : parsed;
  }
  function _parseEnvHide(value) {
    if (value == null) return null;
    const trimmed = String(value).trim();
    if (trimmed === "") return null;
    return trimmed.split(",").map((s) => s.trim()).filter(Boolean);
  }
  function _validateEnum(value, allowed, defaultValue) {
    if (value == null) return defaultValue;
    const lower = String(value).toLowerCase().trim();
    return allowed.includes(lower) ? lower : defaultValue;
  }
  function parseConfig(scriptElement) {
    var _a;
    if (!scriptElement || typeof scriptElement.getAttribute !== "function") {
      return __spreadProps(__spreadValues({}, DEFAULT_CONFIG), { hostPatterns: [...DEFAULT_CONFIG.hostPatterns] });
    }
    return {
      endpoint: scriptElement.getAttribute("data-endpoint") || DEFAULT_CONFIG.endpoint,
      position: _validateEnum(scriptElement.getAttribute("data-position"), VALID_POSITIONS, DEFAULT_CONFIG.position),
      theme: _validateEnum(scriptElement.getAttribute("data-theme"), VALID_THEMES, DEFAULT_CONFIG.theme),
      dismiss: _validateEnum(scriptElement.getAttribute("data-dismiss"), VALID_DISMISS, DEFAULT_CONFIG.dismiss),
      envHide: (_a = _parseEnvHide(scriptElement.getAttribute("data-env-hide"))) != null ? _a : DEFAULT_CONFIG.envHide,
      height: _parseHeight(scriptElement.getAttribute("data-height")),
      debug: _parseBool(scriptElement.getAttribute("data-debug"), DEFAULT_CONFIG.debug),
      poll: _parsePoll(scriptElement.getAttribute("data-poll")),
      push: _parseBool(scriptElement.getAttribute("data-push"), DEFAULT_CONFIG.push),
      token: scriptElement.getAttribute("data-token") || null,
      manual: _parseBool(scriptElement.getAttribute("data-manual"), DEFAULT_CONFIG.manual),
      zIndex: DEFAULT_CONFIG.zIndex,
      hostPatterns: [...DEFAULT_CONFIG.hostPatterns]
    };
  }
  function _validateConfig(config) {
    config.position = _validateEnum(config.position, VALID_POSITIONS, DEFAULT_CONFIG.position);
    config.theme = _validateEnum(config.theme, VALID_THEMES, DEFAULT_CONFIG.theme);
    config.dismiss = _validateEnum(config.dismiss, VALID_DISMISS, DEFAULT_CONFIG.dismiss);
    config.height = _parseHeight(config.height);
    config.poll = _parsePoll(config.poll);
    config.debug = _parseBool(config.debug, DEFAULT_CONFIG.debug);
    config.push = _parseBool(config.push, DEFAULT_CONFIG.push);
    config.manual = _parseBool(config.manual, DEFAULT_CONFIG.manual);
    if (Array.isArray(config.hostPatterns)) {
      config.hostPatterns = [...config.hostPatterns];
    }
    if (Array.isArray(config.envHide)) {
      config.envHide = [...config.envHide];
    }
    return config;
  }
  function resolveConfig(dataAttrs, programmaticOpts) {
    const base = __spreadValues(__spreadValues({}, DEFAULT_CONFIG), dataAttrs);
    if (!programmaticOpts || typeof programmaticOpts !== "object") {
      return _validateConfig(base);
    }
    const merged = __spreadValues({}, base);
    for (const key of Object.keys(programmaticOpts)) {
      if (!Object.hasOwn(DEFAULT_CONFIG, key)) continue;
      merged[key] = programmaticOpts[key];
    }
    if (!merged.endpoint) merged.endpoint = base.endpoint;
    return _validateConfig(merged);
  }

  // src/logger.js
  var LOG_CAP = 20;
  var PREFIX = "[BuildBanner] ";
  function createLogger(debugEnabled) {
    let callCount = 0;
    return {
      /** Emit a diagnostic message, respecting the session cap. */
      log(message) {
        if (callCount >= LOG_CAP) {
          return;
        }
        callCount++;
        const prefixed = PREFIX + message;
        console.debug(prefixed);
        if (debugEnabled) {
          console.warn(prefixed);
        }
      }
    };
  }

  // src/fetch.js
  var TIMEOUT_MS = 3e3;
  async function fetchBannerData(endpoint, options = {}) {
    const { token, logger } = options;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
    try {
      const headers = {};
      if (token) {
        headers["Authorization"] = `Bearer ${token}`;
      }
      const response = await fetch(endpoint, {
        signal: controller.signal,
        headers
      });
      if (!response.ok) {
        clearTimeout(timer);
        if (logger) logger.log(`Fetch failed: HTTP ${response.status}`);
        return null;
      }
      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("text/html")) {
        clearTimeout(timer);
        if (logger) logger.log("Fetch failed: received HTML instead of JSON");
        return null;
      }
      const data = await response.json();
      clearTimeout(timer);
      if (data === null || typeof data !== "object" || Array.isArray(data)) {
        if (logger) logger.log("Fetch failed: response is not a JSON object");
        return null;
      }
      return data;
    } catch (err) {
      clearTimeout(timer);
      const message = err.name === "AbortError" ? "Fetch failed: timeout" : `Fetch failed: ${err.message}`;
      if (logger) logger.log(message);
      return null;
    }
  }

  // src/style-constants.json
  var style_constants_default = {
    DARK_BG: "#1a1a2e",
    DARK_FG: "#e0e0e0",
    DARK_LINK: "#6fa8dc",
    LIGHT_BG: "#f0f0f0",
    LIGHT_FG: "#333333",
    LIGHT_LINK: "#1a5dab",
    FONT_FAMILY: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
    FONT_SIZE: "12px",
    DEFAULT_HEIGHT: 28,
    DEFAULT_Z_INDEX: 999999
  };

  // src/theme.js
  var DARK_BG = style_constants_default.DARK_BG;
  var DARK_FG = style_constants_default.DARK_FG;
  var LIGHT_BG = style_constants_default.LIGHT_BG;
  var LIGHT_FG = style_constants_default.LIGHT_FG;
  var DARK_LINK = style_constants_default.DARK_LINK;
  var LIGHT_LINK = style_constants_default.LIGHT_LINK;
  var FONT_FAMILY = style_constants_default.FONT_FAMILY;
  var FONT_SIZE = style_constants_default.FONT_SIZE;
  function _colorVars(bg, fg, link) {
    return `--bb-bg: ${bg}; --bb-fg: ${fg}; --bb-link: ${link};`;
  }
  function getThemeStyles(theme) {
    if (theme === "light") {
      return `
    :host {
      ${_colorVars(LIGHT_BG, LIGHT_FG, LIGHT_LINK)}
    }`;
    }
    if (theme === "auto") {
      return `
    :host {
      ${_colorVars(DARK_BG, DARK_FG, DARK_LINK)}
    }
    @media (prefers-color-scheme: light) {
      :host {
        ${_colorVars(LIGHT_BG, LIGHT_FG, LIGHT_LINK)}
      }
    }`;
    }
    return `
    :host {
      ${_colorVars(DARK_BG, DARK_FG, DARK_LINK)}
    }`;
  }

  // src/dom.js
  var DEFAULT_HEIGHT = style_constants_default.DEFAULT_HEIGHT;
  var DEFAULT_Z_INDEX = style_constants_default.DEFAULT_Z_INDEX;
  var VALID_POSITION_MODES = ["sticky", "fixed"];
  function _resolveStyleValues(config) {
    const height = parseInt(config.height, 10) || DEFAULT_HEIGHT;
    const zIndex = parseInt(config.zIndex, 10) || DEFAULT_Z_INDEX;
    return { height, zIndex };
  }
  function _buildWrapperCssProperties(height, zIndex, positionMode = "sticky") {
    const safePosition = VALID_POSITION_MODES.includes(positionMode) ? positionMode : "sticky";
    return `
      all: initial;
      display: flex;
      align-items: center;
      gap: 0;
      position: ${safePosition};
      top: 0;
      z-index: ${zIndex};
      height: ${height}px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-family: ${FONT_FAMILY};
      font-size: ${FONT_SIZE};
      line-height: ${height}px;
      color: var(--bb-fg, ${DARK_FG});
      background: var(--bb-bg, ${DARK_BG});
      padding: 0 8px;
      box-sizing: border-box;`;
  }
  function _buildAnchorCss(parentSelector) {
    return `
    ${parentSelector} a {
      color: var(--bb-link);
      text-decoration: none;
    }
    ${parentSelector} a:hover {
      text-decoration: underline;
    }`;
  }
  function _buildStyles(config, positionMode) {
    const { height, zIndex } = _resolveStyleValues(config);
    const theme = config.theme || "dark";
    return `${getThemeStyles(theme)}
    .bb-wrapper {${_buildWrapperCssProperties(height, zIndex, positionMode)}
    }
    .bb-clickable {
      cursor: pointer;
    }
    .bb-live-region {
      display: contents;
    }
    .bb-dismiss {
      all: unset;
      cursor: pointer;
      margin-left: auto;
      padding: 0 4px;
      color: inherit;
      font-size: 14px;
      line-height: inherit;
    }
    .bb-dismiss:focus-visible,
    .bb-wrapper a:focus-visible {
      outline: 2px solid #4da6ff;
      outline-offset: 2px;
      border-radius: 2px;
    }${_buildAnchorCss(".bb-wrapper")}
  `;
  }
  function _buildFallbackStyles(config, positionMode) {
    const { height, zIndex } = _resolveStyleValues(config);
    return `
    .__buildbanner-host {
      all: initial;
      display: block;
    }
    .__buildbanner-wrapper {${_buildWrapperCssProperties(height, zIndex, positionMode)}
      font-weight: normal;
      font-style: normal;
      text-transform: none;
      letter-spacing: normal;
      word-spacing: normal;
      text-align: left;
      text-decoration: none;
      visibility: visible;
      opacity: 1;
      direction: ltr;
    }
    .__buildbanner-clickable {
      cursor: pointer;
    }${_buildAnchorCss(".__buildbanner-wrapper")}
  `;
  }
  function _applyCommonAttributes(host, wrapper) {
    host.setAttribute("data-testid", "buildbanner");
    wrapper.setAttribute("role", "toolbar");
    wrapper.setAttribute("aria-label", "Build information banner");
  }
  function createBannerHost(config = {}, positionMode = "sticky") {
    const logger = createLogger(config.debug);
    if (!document.body) {
      logger.log("document.body is null \u2014 cannot create banner host");
      return null;
    }
    const hasShadow = typeof HTMLElement.prototype.attachShadow === "function";
    let host;
    let shadowRoot = null;
    let wrapper;
    let fallbackStyle = null;
    if (hasShadow) {
      host = document.createElement("build-banner");
      shadowRoot = host.attachShadow({ mode: "open" });
      const style = document.createElement("style");
      style.textContent = _buildStyles(config, positionMode);
      shadowRoot.appendChild(style);
      wrapper = document.createElement("div");
      wrapper.className = "bb-wrapper";
      shadowRoot.appendChild(wrapper);
    } else {
      host = document.createElement("div");
      host.className = "__buildbanner-host";
      fallbackStyle = document.createElement("style");
      fallbackStyle.textContent = _buildFallbackStyles(config, positionMode);
      document.head.appendChild(fallbackStyle);
      wrapper = document.createElement("div");
      wrapper.className = "__buildbanner-wrapper";
      host.appendChild(wrapper);
    }
    _applyCommonAttributes(host, wrapper);
    if (config.position === "bottom") {
      document.body.appendChild(host);
    } else {
      document.body.insertBefore(host, document.body.firstChild);
    }
    return { host, shadowRoot, wrapper, fallbackStyle };
  }
  function destroyBannerHost(host, fallbackStyle) {
    if (host && host.parentNode) {
      host.parentNode.removeChild(host);
    }
    if (fallbackStyle && fallbackStyle.parentNode) {
      fallbackStyle.parentNode.removeChild(fallbackStyle);
    }
  }

  // src/time.js
  function _formatElapsed(ms) {
    const totalSeconds = Math.floor(ms / 1e3);
    const days = Math.floor(totalSeconds / 86400);
    const hours = Math.floor(totalSeconds % 86400 / 3600);
    const minutes = Math.floor(totalSeconds % 3600 / 60);
    const seconds = totalSeconds % 60;
    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m`;
    return `${seconds}s`;
  }
  function _parseElapsedMs(isoString) {
    if (!isoString) return null;
    const date = new Date(isoString);
    if (isNaN(date.getTime())) return null;
    const elapsed = Date.now() - date.getTime();
    if (elapsed < 0) return null;
    return elapsed;
  }
  function formatUptime(serverStartedISO) {
    const elapsed = _parseElapsedMs(serverStartedISO);
    if (elapsed === null) return null;
    return `up ${_formatElapsed(elapsed)}`;
  }
  function formatDeployAge(deployedAtISO) {
    const elapsed = _parseElapsedMs(deployedAtISO);
    if (elapsed === null) return null;
    return `deployed ${_formatElapsed(elapsed)} ago`;
  }
  function startUptimeTicker(element, serverStartedISO) {
    if (!element || !serverStartedISO) return null;
    const start = new Date(serverStartedISO);
    if (isNaN(start.getTime())) return null;
    const initialElapsed = Date.now() - start.getTime();
    if (initialElapsed >= 0) {
      element.textContent = `up ${_formatElapsed(initialElapsed)}`;
    }
    const timerId = setInterval(() => {
      if (!element.isConnected) {
        clearInterval(timerId);
        return;
      }
      const elapsed = Date.now() - start.getTime();
      if (elapsed >= 0) {
        element.textContent = `up ${_formatElapsed(elapsed)}`;
      }
    }, 6e4);
    return timerId;
  }

  // src/links.js
  var BUILTIN_HOSTS = {
    "github.com": { commitPath: "/commit/{sha}", treePath: "/tree/{branch}" },
    "gitlab.com": { commitPath: "/-/commit/{sha}", treePath: "/-/tree/{branch}" },
    "bitbucket.org": { commitPath: "/commits/{sha}", treePath: "/src/{branch}" }
  };
  function _findPattern(hostname, hostPatterns) {
    for (const pattern of hostPatterns) {
      if (pattern.host === hostname) return pattern;
    }
    return BUILTIN_HOSTS[hostname] || null;
  }
  var TEMPLATE_MAP = { commit: "commitPath", tree: "treePath" };
  function _buildUrl(repoUrl, template, value) {
    const encoded = encodeURIComponent(value);
    const path = template.replace("{sha}", encoded).replace("{branch}", encoded);
    return repoUrl.replace(/\/+$/, "") + path;
  }
  function createLink(repoUrl, type, value, hostPatterns = []) {
    if (!repoUrl || !value) return null;
    if (!Array.isArray(hostPatterns)) return null;
    const templateKey = TEMPLATE_MAP[type];
    if (!templateKey) return null;
    let hostname;
    try {
      hostname = new URL(repoUrl).hostname;
    } catch (e) {
      return null;
    }
    const pattern = _findPattern(hostname, hostPatterns);
    if (!pattern) return null;
    const template = pattern[templateKey];
    if (!template) return null;
    return _buildUrl(repoUrl, template, value);
  }

  // src/clipboard.js
  var COPIED_DISPLAY_MS = 1500;
  function _execCommandCopy(text) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (e) {
      ok = false;
    }
    document.body.removeChild(textarea);
    return ok;
  }
  function _flashCopied(element, originalText) {
    element.textContent = "Copied!";
    return setTimeout(() => {
      element.textContent = originalText;
    }, COPIED_DISPLAY_MS);
  }
  function attachCopyHandler(shaElement, fullSha, logger) {
    let isCopied = false;
    shaElement.addEventListener("click", (e) => {
      if (e.metaKey || e.ctrlKey) return;
      e.preventDefault();
      if (isCopied) return;
      const originalText = shaElement.textContent;
      isCopied = true;
      const onSuccess = () => {
        _flashCopied(shaElement, originalText);
        setTimeout(() => {
          isCopied = false;
        }, COPIED_DISPLAY_MS);
      };
      const onFailure = () => {
        isCopied = false;
        logger.log("clipboard copy failed");
      };
      if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
        navigator.clipboard.writeText(fullSha).then(onSuccess, () => {
          if (_execCommandCopy(fullSha)) {
            onSuccess();
          } else {
            onFailure();
          }
        });
      } else if (_execCommandCopy(fullSha)) {
        onSuccess();
      } else {
        onFailure();
      }
    });
  }

  // src/segments.js
  var STATUS_DOTS = {
    pass: "\u{1F7E2}",
    fresh: "\u{1F7E2}",
    fail: "\u{1F534}",
    stale: "\u{1F534}",
    running: "\u{1F7E1}",
    building: "\u{1F7E1}",
    idle: "\u26AA"
  };
  var DEFAULT_DOT = "\u26AA";
  var ALLOWED_PROTOCOLS = /* @__PURE__ */ new Set(["http:", "https:"]);
  function _getStatusDot(status) {
    if (!status) return DEFAULT_DOT;
    return STATUS_DOTS[status.toLowerCase()] || DEFAULT_DOT;
  }
  function _createSeparator() {
    const sep = document.createElement("span");
    sep.textContent = " \xB7 ";
    return sep;
  }
  function _createSpan(segmentName, text) {
    const span = document.createElement("span");
    span.setAttribute("data-segment", segmentName);
    span.textContent = text;
    return span;
  }
  function _isSafeUrl(url) {
    try {
      const parsed = new URL(url, window.location.origin);
      return ALLOWED_PROTOCOLS.has(parsed.protocol);
    } catch (e) {
      return false;
    }
  }
  function _createStatusSegment(segmentName, statusObj) {
    if (!statusObj || !statusObj.status) return null;
    const dot = _getStatusDot(statusObj.status);
    const text = statusObj.summary ? `${dot} ${statusObj.summary}` : `${dot} ${statusObj.status}`;
    if (statusObj.url && _isSafeUrl(statusObj.url)) {
      const anchor = document.createElement("a");
      anchor.setAttribute("data-segment", segmentName);
      anchor.href = statusObj.url;
      anchor.target = "_blank";
      anchor.rel = "noopener";
      anchor.textContent = text;
      return anchor;
    }
    return _createSpan(segmentName, text);
  }
  function _createLinkedSpan(segmentName, text, href) {
    const anchor = document.createElement("a");
    anchor.setAttribute("data-segment", segmentName);
    anchor.href = href;
    anchor.target = "_blank";
    anchor.rel = "noopener";
    anchor.textContent = text;
    return anchor;
  }
  function _isBranchHidden(branch) {
    return !branch || branch === "HEAD";
  }
  function _formatCommitDate(isoString) {
    if (!isoString) return null;
    const date = new Date(isoString);
    if (isNaN(date.getTime())) return null;
    return date.toLocaleString();
  }
  function _appendSegment(wrapper, segment, isFirst) {
    if (!isFirst) {
      wrapper.appendChild(_createSeparator());
    }
    wrapper.appendChild(segment);
  }
  function _createMaybeLinkedSegment(segmentName, text, linkUrl) {
    return linkUrl ? _createLinkedSpan(segmentName, text, linkUrl) : _createSpan(segmentName, text);
  }
  function _hasStatusChanged(field, newStatus, previousStatuses) {
    const current = newStatus && newStatus.status ? newStatus.status : null;
    const isFirstTrack = !(field in previousStatuses);
    const prev = previousStatuses[field];
    previousStatuses[field] = current;
    if (isFirstTrack) return true;
    return prev !== current;
  }
  function _buildStatusContainer(testsSegment, buildSegment, isLive) {
    const container = document.createElement("div");
    container.className = "bb-live-region";
    container.setAttribute("data-bb-live-region", "");
    if (isLive) {
      container.setAttribute("role", "status");
      container.setAttribute("aria-live", "polite");
    }
    let isFirst = true;
    if (testsSegment) {
      _appendSegment(container, testsSegment, isFirst);
      isFirst = false;
    }
    if (buildSegment) {
      _appendSegment(container, buildSegment, isFirst);
    }
    return container;
  }
  function renderSegments(data, wrapper, config = {}, previousStatuses = {}) {
    const segments = [];
    let tickerTimerId = null;
    const hostPatterns = config.hostPatterns || [];
    if (data.app_name) {
      segments.push(_createSpan("app-name", data.app_name));
    }
    if (data.environment) {
      segments.push(_createSpan("environment", data.environment));
    }
    if (!_isBranchHidden(data.branch)) {
      const branchLink = createLink(data.repo_url, "tree", data.branch, hostPatterns);
      segments.push(_createMaybeLinkedSegment("branch", data.branch, branchLink));
    }
    if (data.sha) {
      const shaValue = data.sha_full || data.sha;
      const shaLink = createLink(data.repo_url, "commit", shaValue, hostPatterns);
      const shaEl = _createMaybeLinkedSegment("sha", data.sha, shaLink);
      shaEl.classList.add("bb-clickable");
      const logger = createLogger(config.debug);
      attachCopyHandler(shaEl, shaValue, logger);
      segments.push(shaEl);
    }
    const commitDateStr = _formatCommitDate(data.commit_date);
    if (commitDateStr) {
      segments.push(_createSpan("commit-date", commitDateStr));
    }
    const uptimeStr = formatUptime(data.server_started);
    if (uptimeStr) {
      const uptimeSpan = _createSpan("uptime", uptimeStr);
      segments.push(uptimeSpan);
      tickerTimerId = startUptimeTicker(uptimeSpan, data.server_started);
    }
    const deployAgeStr = formatDeployAge(data.deployed_at);
    if (deployAgeStr) {
      segments.push(_createSpan("deploy-age", deployAgeStr));
    }
    const testsSegment = _createStatusSegment("tests", data.tests);
    const buildSegment = _createStatusSegment("build", data.build);
    const hasStatusSegments = testsSegment || buildSegment;
    if (hasStatusSegments) {
      const testsChanged = _hasStatusChanged("tests", data.tests, previousStatuses);
      const buildChanged = _hasStatusChanged("build", data.build, previousStatuses);
      const isLive = testsChanged || buildChanged;
      segments.push(_buildStatusContainer(testsSegment, buildSegment, isLive));
    }
    if (data.port !== null && data.port !== void 0) {
      segments.push(_createSpan("port", String(data.port)));
    }
    if (data.custom && typeof data.custom === "object") {
      const keys = Object.keys(data.custom).sort();
      for (const key of keys) {
        const value = data.custom[key];
        if (typeof value === "string") {
          segments.push(_createSpan(`custom-${key}`, value));
        }
      }
    }
    for (let i = 0; i < segments.length; i++) {
      _appendSegment(wrapper, segments[i], i === 0);
    }
    return { tickerTimerId };
  }

  // src/token-warnings.js
  var SHORT_TOKEN_THRESHOLD = 16;
  var SAFE_SUFFIXES = [".local", ".internal", ".test"];
  var RFC1918_PREFIXES = ["10.", "192.168."];
  function _isSafeHostname(hostname) {
    if (hostname === "localhost" || hostname === "127.0.0.1" || hostname === "[::1]") {
      return true;
    }
    if (RFC1918_PREFIXES.some((prefix) => hostname.startsWith(prefix))) return true;
    if (/^172\.(1[6-9]|2\d|3[01])\./.test(hostname)) return true;
    return SAFE_SUFFIXES.some((suffix) => hostname.endsWith(suffix));
  }
  function checkTokenWarnings(config) {
    try {
      if (config.token == null) return;
      if (config.token === "" || config.token.length < SHORT_TOKEN_THRESHOLD) {
        console.warn(
          "[BuildBanner] Token is shorter than 16 characters. Short tokens offer minimal protection."
        );
      }
      if (typeof window !== "undefined" && window.location.protocol === "https:" && !_isSafeHostname(window.location.hostname)) {
        console.warn(
          "[BuildBanner] Token auth detected on a public-facing origin. data-token is intended for staging/internal use only."
        );
      }
    } catch (e) {
    }
  }

  // src/dismiss.js
  var STORAGE_KEY = "buildbanner-dismissed";
  var dismissedInMemory = false;
  function _getStorage(dismiss) {
    try {
      if (dismiss === "session") return sessionStorage;
      if (dismiss === "permanent") return localStorage;
    } catch (e) {
    }
    return null;
  }
  function isDismissed(config) {
    if (!config || config.dismiss === "none") return false;
    const storage = _getStorage(config.dismiss);
    if (storage) {
      try {
        if (storage.getItem(STORAGE_KEY) !== null) return true;
      } catch (e) {
      }
    }
    return dismissedInMemory;
  }
  function createDismissButton(config, onDismiss) {
    if (!config || config.dismiss === "none") return null;
    const button = document.createElement("button");
    button.textContent = "\u2715";
    button.setAttribute("aria-label", "Close build banner");
    button.className = "bb-dismiss";
    button.addEventListener("click", () => {
      let wrote = false;
      const storage = _getStorage(config.dismiss);
      if (storage) {
        try {
          storage.setItem(STORAGE_KEY, "1");
          wrote = true;
        } catch (e) {
        }
      }
      if (!wrote) {
        dismissedInMemory = true;
      }
      if (typeof onDismiss === "function") {
        onDismiss();
      }
    });
    return button;
  }
  function resetDismiss() {
    dismissedInMemory = false;
  }

  // src/polling.js
  var MAX_INTERVAL_SEC = 300;
  function startPolling(config, fetchFn, onData, logger) {
    const baseInterval = config.poll;
    if (!baseInterval || baseInterval <= 0) return null;
    const state = {
      timerId: null,
      listenerRef: null,
      currentInterval: baseInterval,
      stopped: false
    };
    const _tick = async () => {
      try {
        const data = await fetchFn();
        if (data) {
          state.currentInterval = baseInterval;
          onData(data);
        } else {
          _handleFailure(state, logger, "Poll failed");
        }
      } catch (err) {
        _handleFailure(state, logger, `Poll error: ${err.message}`);
      }
      if (!state.stopped) {
        _scheduleNext(state, _tick);
      }
    };
    const _onVisibilityChange = () => {
      if (state.stopped) return;
      if (document.hidden) {
        _clearTimer(state);
      } else {
        _clearTimer(state);
        _tick();
      }
    };
    state.listenerRef = _onVisibilityChange;
    document.addEventListener("visibilitychange", _onVisibilityChange);
    _scheduleNext(state, _tick);
    return state;
  }
  function stopPolling(state) {
    if (!state) return;
    state.stopped = true;
    _clearTimer(state);
    if (state.listenerRef) {
      document.removeEventListener("visibilitychange", state.listenerRef);
      state.listenerRef = null;
    }
  }
  function _handleFailure(state, logger, reason) {
    _backoff(state);
    if (logger) logger.log(`${reason}, backing off to ${state.currentInterval}s`);
  }
  function _backoff(state) {
    const doubled = state.currentInterval * 2;
    state.currentInterval = Math.min(doubled, MAX_INTERVAL_SEC);
  }
  function _scheduleNext(state, tickFn) {
    state.timerId = setTimeout(tickFn, state.currentInterval * 1e3);
  }
  function _clearTimer(state) {
    if (state.timerId != null) {
      clearTimeout(state.timerId);
      state.timerId = null;
    }
  }

  // src/push.js
  function applyPush(config, bannerHeight, logger) {
    const prop = _paddingProperty(config);
    const existing = _readPadding(prop);
    if (!config.push) {
      return { mode: "overlay", originalPadding: existing };
    }
    if (existing !== 0) {
      if (logger) {
        logger.log("Push mode fell back to overlay due to existing padding");
      }
      return { mode: "overlay", originalPadding: existing };
    }
    document.documentElement.style[prop] = `${bannerHeight}px`;
    return { mode: "push", originalPadding: 0 };
  }
  function removePush(bannerHeight, pushState, config) {
    if (!pushState || pushState.mode !== "push") return;
    const prop = _paddingProperty(config);
    const current = _readPadding(prop);
    const expected = pushState.originalPadding + bannerHeight;
    if (current === expected) {
      document.documentElement.style[prop] = "";
    } else {
      const restored = Math.max(0, current - bannerHeight);
      document.documentElement.style[prop] = restored ? `${restored}px` : "";
    }
  }
  function resolvePositionMode(pushMode) {
    return pushMode === "push" ? "sticky" : "fixed";
  }
  function _paddingProperty(config) {
    return config && config.position === "bottom" ? "paddingBottom" : "paddingTop";
  }
  function _readPadding(prop) {
    const raw = getComputedStyle(document.documentElement)[prop];
    return parseInt(raw, 10) || 0;
  }

  // src/env-hide.js
  function shouldHide(envHideList, environment) {
    if (!Array.isArray(envHideList) || envHideList.length === 0) return false;
    if (!environment) return false;
    const lowerEnv = String(environment).toLowerCase();
    return envHideList.some((entry) => String(entry).toLowerCase() === lowerEnv);
  }

  // src/main.js
  var SYMBOL_KEY = Symbol.for("buildbanner");
  function _getInstance() {
    return window[SYMBOL_KEY];
  }
  function _getActiveInstance() {
    const instance = _getInstance();
    if (!instance || instance.destroyed) return null;
    return instance;
  }
  function _setInstance(instance) {
    window[SYMBOL_KEY] = instance;
  }
  function _clearInstance() {
    window[SYMBOL_KEY] = null;
  }
  function _teardown(instance) {
    if (instance.tickerTimerId) {
      clearInterval(instance.tickerTimerId);
    }
    if (instance.pollingState) {
      stopPolling(instance.pollingState);
    }
    removePush(instance.bannerHeight, instance.pushState, instance.config);
    destroyBannerHost(instance.host, instance.fallbackStyle);
  }
  async function init(opts = {}) {
    try {
      const existing = _getInstance();
      if (existing && !existing.destroyed) {
        if (!existing.pending) {
          console.debug("[BuildBanner] Already initialized, skipping");
        }
        return;
      }
      const pending = { destroyed: false, pending: true };
      _setInstance(pending);
      const config = resolveConfig({}, opts);
      checkTokenWarnings(config);
      if (isDismissed(config)) {
        pending.destroyed = true;
        return;
      }
      const logger = createLogger(config.debug);
      const data = await fetchBannerData(config.endpoint, {
        token: config.token,
        logger
      });
      if (!data) {
        _clearInstance();
        return;
      }
      if (Array.isArray(config.envHide) && config.envHide.length > 0 && !data.environment) {
        logger.log("envHide is configured but server response has no environment field");
      }
      if (shouldHide(config.envHide, data.environment)) {
        logger.log("Banner hidden: environment '" + data.environment + "' is in envHide list");
        pending.destroyed = true;
        _clearInstance();
        return;
      }
      const bannerHeight = parseInt(config.height, 10) || DEFAULT_HEIGHT;
      const pushState = applyPush(config, bannerHeight, logger);
      const positionMode = resolvePositionMode(pushState.mode);
      const result = createBannerHost(config, positionMode);
      if (!result) {
        removePush(bannerHeight, pushState, config);
        _clearInstance();
        return;
      }
      const { host, shadowRoot, wrapper, fallbackStyle } = result;
      const previousStatuses = {};
      const { tickerTimerId } = renderSegments(data, wrapper, config, previousStatuses);
      const instance = {
        host,
        shadowRoot,
        wrapper,
        fallbackStyle,
        tickerTimerId,
        pollingState: null,
        destroyed: false,
        pushState,
        bannerHeight,
        config,
        data,
        previousStatuses
      };
      if (config.poll > 0) {
        const pollFetchFn = () => fetchBannerData(config.endpoint, { token: config.token, logger });
        const pollOnData = (newData) => {
          instance.data = newData;
          _rerender(instance);
        };
        instance.pollingState = startPolling(config, pollFetchFn, pollOnData, logger);
      }
      const dismissBtn = createDismissButton(config, () => {
        _teardown(instance);
        _clearInstance();
      });
      if (dismissBtn) {
        wrapper.appendChild(dismissBtn);
      }
      _setInstance(instance);
    } catch (err) {
      _clearInstance();
      console.debug("[BuildBanner] init failed:", err);
    }
  }
  function _rerender(instance) {
    if (instance.tickerTimerId) {
      clearInterval(instance.tickerTimerId);
    }
    instance.wrapper.textContent = "";
    const rendered = renderSegments(
      instance.data,
      instance.wrapper,
      instance.config,
      instance.previousStatuses
    );
    instance.tickerTimerId = rendered.tickerTimerId;
  }
  async function refresh() {
    try {
      const instance = _getActiveInstance();
      if (!instance) return;
      const logger = createLogger(instance.config.debug);
      const newData = await fetchBannerData(instance.config.endpoint, {
        token: instance.config.token,
        logger,
        isRefetch: true
      });
      if (!newData) return;
      instance.data = newData;
      _rerender(instance);
    } catch (err) {
      console.debug("[BuildBanner] refresh failed:", err);
    }
  }
  function update(partialData) {
    try {
      const instance = _getActiveInstance();
      if (!instance || !instance.data) return;
      if (!partialData || typeof partialData !== "object") return;
      if (partialData.custom && instance.data.custom) {
        partialData = __spreadProps(__spreadValues({}, partialData), {
          custom: __spreadValues(__spreadValues({}, instance.data.custom), partialData.custom)
        });
      }
      instance.data = __spreadValues(__spreadValues({}, instance.data), partialData);
      _rerender(instance);
    } catch (err) {
      console.debug("[BuildBanner] update failed:", err);
    }
  }
  function destroy() {
    try {
      const instance = _getInstance();
      if (!instance) return;
      _teardown(instance);
      resetDismiss();
      instance.destroyed = true;
      _clearInstance();
      _disableMethods();
    } catch (err) {
      console.debug("[BuildBanner] destroy failed:", err);
    }
  }
  function isVisible() {
    try {
      return Boolean(_getActiveInstance());
    } catch (e) {
      return false;
    }
  }
  var ORIGINAL_METHODS = { init, destroy, refresh, update, isVisible };
  function _disableMethods() {
    BuildBanner.destroy = () => {
    };
    BuildBanner.refresh = () => Promise.resolve();
    BuildBanner.update = () => {
    };
    BuildBanner.isVisible = () => false;
    BuildBanner.init = function restoreAndInit(opts) {
      _restoreMethods();
      return init(opts);
    };
  }
  function _restoreMethods() {
    Object.assign(BuildBanner, ORIGINAL_METHODS);
  }
  function _autoInit() {
    const scripts = document.querySelectorAll("script[src]");
    let scriptEl = null;
    for (const s of scripts) {
      if (s.src && s.src.includes("buildbanner")) {
        scriptEl = s;
        break;
      }
    }
    if (!scriptEl) return;
    if (scriptEl.dataset.manual !== void 0) return;
    const config = parseConfig(scriptEl);
    init(config).catch(() => {
    });
  }
  if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", _autoInit);
    } else {
      _autoInit();
    }
  }
  var BuildBanner = __spreadValues({}, ORIGINAL_METHODS);
  if (typeof window !== "undefined") {
    window.BuildBanner = BuildBanner;
  }
  var main_default = BuildBanner;
})();
