/**
 * Gallery UI for browsing and selecting visualizations from manifest.json.
 * Renders thumbnail cards grouped by category with filtering support.
 */

/* ── Constants ── */
const PLACEHOLDER_THUMB = 'data:image/svg+xml,' + encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" width="160" height="120" fill="%231a1a2e">' +
  '<rect width="160" height="120"/>' +
  '<text x="80" y="60" text-anchor="middle" fill="%23666" font-size="14">No thumbnail</text></svg>',
);
const ALL_CATEGORY = 'All';

/**
 * Build the gallery from a manifest array and wire interactions.
 * @param {HTMLElement} galleryContainer - The gallery panel element.
 * @param {Array} items - Manifest entries with name, category, display_name, etc.
 * @param {function} onSelect - Callback when a card is clicked: (item) => void.
 * @returns {{selectByName: function, getItems: function}}
 */
export function buildGallery(galleryContainer, items, onSelect) {
  const filterBar = galleryContainer.querySelector('#gallery-filter-bar');
  const grid = galleryContainer.querySelector('#gallery-grid');

  const categories = _extractCategories(items);
  _renderFilterBar(filterBar, categories);
  _renderCards(grid, items);

  let activeCategory = ALL_CATEGORY;
  let selectedName = null;

  /* ── Filter bar clicks ── */
  filterBar.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-category]');
    if (!btn) return;
    activeCategory = btn.dataset.category;
    _updateFilterHighlight(filterBar, activeCategory);
    _filterCards(grid, activeCategory);
  });

  /* ── Card clicks ── */
  grid.addEventListener('click', (e) => {
    const card = e.target.closest('.gallery-card');
    if (!card) return;
    const name = card.dataset.name;
    if (name === selectedName) return;
    const item = items.find((it) => it.name === name);
    if (!item) return;
    _highlightCard(grid, name);
    selectedName = name;
    onSelect(item);
  });

  /** Select a card by name programmatically. */
  function selectByName(name) {
    const item = items.find((it) => it.name === name);
    if (!item) return false;
    _highlightCard(grid, name);
    selectedName = name;
    /* Ensure the card's category is visible */
    if (activeCategory !== ALL_CATEGORY && item.category !== activeCategory) {
      activeCategory = ALL_CATEGORY;
      _updateFilterHighlight(filterBar, activeCategory);
      _filterCards(grid, activeCategory);
    }
    _scrollCardIntoView(grid, name);
    onSelect(item);
    return true;
  }

  function getItems() {
    return items;
  }

  return {selectByName, getItems};
}

/**
 * Parse the query string for a `name` parameter.
 * @returns {string|null}
 */
export function getQueryParamName() {
  const params = new URLSearchParams(window.location.search);
  return params.get('name') || null;
}

/**
 * Resolve mesh and cloud paths for a manifest item.
 * Falls back to ./data/{name}/ convention.
 */
export function resolveItemPaths(item) {
  const basePath = './data/' + item.name;
  return {
    mesh: item.mesh || (basePath + '/mesh.glb'),
    cloud: item.cloud || (basePath + '/cloud.ply'),
  };
}

/* ── Private helpers ── */

function _extractCategories(items) {
  const set = new Set();
  for (const item of items) {
    if (item.category) set.add(item.category);
  }
  return [ALL_CATEGORY, ...Array.from(set).sort()];
}

function _renderFilterBar(container, categories) {
  container.innerHTML = '';
  for (const cat of categories) {
    const btn = document.createElement('button');
    btn.className = 'gallery-filter-btn' + (cat === ALL_CATEGORY ? ' active' : '');
    btn.dataset.category = cat;
    btn.textContent = cat;
    container.appendChild(btn);
  }
}

function _renderCards(grid, items) {
  grid.innerHTML = '';
  for (const item of items) {
    const card = document.createElement('div');
    card.className = 'gallery-card';
    card.dataset.name = item.name;
    card.dataset.category = item.category || '';

    const thumb = document.createElement('img');
    thumb.className = 'gallery-thumb';
    thumb.src = item.thumbnail || PLACEHOLDER_THUMB;
    thumb.alt = item.display_name || item.name;
    thumb.loading = 'lazy';

    const body = document.createElement('div');
    body.className = 'gallery-card-body';

    const title = document.createElement('div');
    title.className = 'gallery-card-title';
    title.textContent = item.display_name || item.name;

    body.appendChild(title);

    if (item.category) {
      const badge = document.createElement('span');
      badge.className = 'gallery-category-badge';
      badge.textContent = item.category;
      body.appendChild(badge);
    }

    if (item.description) {
      const desc = document.createElement('div');
      desc.className = 'gallery-card-desc';
      desc.textContent = item.description;
      body.appendChild(desc);
    }

    card.appendChild(thumb);
    card.appendChild(body);
    grid.appendChild(card);
  }
}

function _updateFilterHighlight(filterBar, activeCategory) {
  for (const btn of filterBar.children) {
    btn.classList.toggle('active', btn.dataset.category === activeCategory);
  }
}

function _filterCards(grid, category) {
  for (const card of grid.children) {
    if (category === ALL_CATEGORY) {
      card.style.display = '';
    } else {
      card.style.display = card.dataset.category === category ? '' : 'none';
    }
  }
}

function _highlightCard(grid, name) {
  for (const card of grid.children) {
    card.classList.toggle('selected', card.dataset.name === name);
  }
}

function _scrollCardIntoView(grid, name) {
  const card = grid.querySelector(`.gallery-card[data-name="${CSS.escape(name)}"]`);
  if (card) card.scrollIntoView({behavior: 'smooth', block: 'nearest'});
}
