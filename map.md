---
layout: default
title: Interactive state map
---

# State-level suicide burden + recommended interventions

<link rel="stylesheet"
      href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
      crossorigin="anonymous" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
        crossorigin="anonymous"></script>
<style>#map { height: 600px; width: 100%; }</style>

<div id="map"></div>

<script>
// India states map with NCRB 2023 suicide rates + click-to-recommend
const map = L.map('map').setView([22.0, 79.0], 5);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 8, attribution: '&copy; OpenStreetMap'
}).addTo(map);

// State suicide rates (NCRB 2023 from data/state_rates.json)
fetch('data/state_rates.json').then(r => r.json()).then(rates => {
  fetch('data/india_states.geojson').then(r => r.json()).then(geo => {
    L.geoJSON(geo, {
      style: feature => {
        const name = feature.properties.NAME_1 || feature.properties.st_nm;
        const rate = rates[name] || 0;
        return {
          fillColor: rate > 25 ? '#67000d' : rate > 18 ? '#cb181d' : rate > 12 ? '#fb6a4a' : rate > 6 ? '#fcae91' : '#fee5d9',
          weight: 1, opacity: 1, color: 'white', fillOpacity: 0.7
        };
      },
      onEachFeature: (feature, layer) => {
        const name = feature.properties.NAME_1 || feature.properties.st_nm;
        const rate = rates[name] || 'NA';
        layer.bindPopup(`<b>${name}</b><br/>NCRB suicide rate 2023: ${rate}/100k<br/>Top recommendation: see intervention list`);
      }
    }).addTo(map);
  });
});
</script>

## How to read the map

- Colour gradient: light orange (low rate) -> deep red (high rate)
- Click any state for the NCRB 2023 rate
- Top 5 priority interventions (TOPSIS equal-weights): [view list](../#top-5-interventions-topsis-equal-weights)
