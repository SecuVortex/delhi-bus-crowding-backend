#!/usr/bin/env python3
"""
backend/api.py - Delhi Bus Crowding Platform
Fast in-memory routing: loads precomputed tables at startup for <150ms latency
Render-optimized: 256MB memory cap, CORS, $PORT support
"""
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import duckdb
from math import radians, sin, cos, sqrt, atan2
import time
import os

app = FastAPI()

# --- CORS --- allow all origins for free tier compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = 'data/processed/crowding.duckdb'
HEALTH_TTL_SECONDS = 300

# Bus capacities for load ratio calculation
BUS_CAPACITIES = {'auto': 45, 'small': 30, 'medium': 60, 'large': 90}

# Delhi geographic bounds (strictly inside these limits)
DELHI_LAT_MIN, DELHI_LAT_MAX = 28.40, 28.90
DELHI_LNG_MIN, DELHI_LNG_MAX = 76.80, 77.40

# Render port support
RENDER_PORT = int(os.environ.get("PORT", 8000))

# ============================================================================
# GLOBAL IN-MEMORY DATA LOADED AT STARTUP
# ============================================================================

_STOP_COORDS = {}           # stop_name -> (lat, lng) - valid Delhi coords strictly inside bounds
_ROUTE_STOPS = {}           # route_id -> [(stop_name, stop_sequence), ...]
_STOP_LOOKUP = {}           # (route_id, stop_sequence) -> stop_name
_CROWDING = {}              # (route_id, stop_sequence, day_type, hour_bin) -> dict
_STOP_SET = set()           # all stop names
_HOTSPOTS = []              # top 25 overloaded stretches with valid coords
_STOP_ROUTES = {}           # stop_name -> set of route_ids (for transfer lookups)

# ============================================================================
# COORDINATE CLAMPING HELPER
# ============================================================================

def clamp_coords(lat, lng):
    """Clamp coordinates to valid Delhi bounds [28.40-28.90 N, 76.80-77.40 E]"""
    if lat is None or lng is None:
        return (28.63, 77.22)  # Default Delhi center
    lat = max(DELHI_LAT_MIN, min(DELHI_LAT_MAX, lat))
    lng = max(DELHI_LNG_MIN, min(DELHI_LNG_MAX, lng))
    return (lat, lng)

# ============================================================================
# LOAD DATA AT STARTUP (with memory caps for Render 512 MB tier)
# ============================================================================

def load_data():
    global _STOP_COORDS, _ROUTE_STOPS, _STOP_LOOKUP, _CROWDING, _STOP_SET, _HOTSPOTS, _STOP_ROUTES
    con = duckdb.connect(DB_PATH, read_only=True)
    
    # --- Render memory cap: never exceed 256 MB ---
    con.execute("SET memory_limit='256MB'; SET threads=1;")
    
    # Build stop coordinates from fact_stretch_crowding (has real coords)
    stop_coords_dict = {}
    
    crowd_df = con.execute("""
        SELECT DISTINCT from_stop_name, from_lat, from_lng, to_stop_name, to_lat, to_lng
        FROM fact_stretch_crowding
        WHERE from_lat IS NOT NULL AND from_lng IS NOT NULL
    """).fetchdf()
    
    for _, row in crowd_df.iterrows():
        if row['from_stop_name']:
            if row['from_stop_name'] not in stop_coords_dict:
                stop_coords_dict[row['from_stop_name']] = (row['from_lat'], row['from_lng'])
        if row['to_stop_name']:
            if row['to_stop_name'] not in stop_coords_dict:
                stop_coords_dict[row['to_stop_name']] = (row['to_lat'], row['to_lng'])
    
    # Also load from dim_stops for any missing stops
    stops_df = con.execute("SELECT stop_name, lat, lng FROM dim_stops").fetchdf()
    for _, row in stops_df.iterrows():
        if row['stop_name'] not in stop_coords_dict:
            stop_coords_dict[row['stop_name']] = (row['lat'], row['lng'])
    
    # Clamp all stop coordinates to Delhi bounds
    _stop_coords_dict = {}
    for stop_name, coords in stop_coords_dict.items():
        lat, lng = coords
        _stop_coords_dict[stop_name] = clamp_coords(lat, lng)
    
    _STOP_COORDS = _stop_coords_dict
    _STOP_SET = set(_STOP_COORDS.keys())
    
    # Load dim_route_stops_out (route sequences)
    route_stops_df = con.execute("""
        SELECT route_id, stop_name, stop_sequence
        FROM dim_route_stops_out
        WHERE stop_name IN (SELECT stop_name FROM dim_stops WHERE stop_name IS NOT NULL)
        ORDER BY route_id, stop_sequence
    """).fetchdf()
    _ROUTE_STOPS = {}
    _STOP_LOOKUP = {}
    for _, row in route_stops_df.iterrows():
        route_id = row['route_id']
        if route_id not in _ROUTE_STOPS:
            _ROUTE_STOPS[route_id] = []
        _ROUTE_STOPS[route_id].append((row['stop_name'], row['stop_sequence']))
        _STOP_LOOKUP[(route_id, row['stop_sequence'])] = row['stop_name']
    
    # Build _STOP_ROUTES for transfer lookups
    _STOP_ROUTES = {}
    for route_id, stops in _ROUTE_STOPS.items():
        for stop_name, _ in stops:
            if stop_name not in _STOP_ROUTES:
                _STOP_ROUTES[stop_name] = set()
            _STOP_ROUTES[stop_name].add(route_id)
    
    # Load fact_stretch_crowding (crowding data) - indexed by (route_id, stop_sequence, day_type, hour_bin)
    crowd_df = con.execute("SELECT * FROM fact_stretch_crowding").fetchdf()
    for _, row in crowd_df.iterrows():
        from_lat = row.get('from_lat', 28.63)
        from_lng = row.get('from_lng', 77.22)
        to_lat = row.get('to_lat', 28.63)
        to_lng = row.get('to_lng', 77.22)
        median_load = row.get('median_observed_load', 0)
        dist_km = row.get('distance_km', 0)
        conf = row.get('confidence', 'Low')
        load_ratio_label = row.get('load_ratio_label', 'SEATS_AVAILABLE')
        
        key = (row['route_id'], row['stop_sequence'], row['day_type'], row['hour_bin'])
        from_lat2, from_lng2 = clamp_coords(from_lat, from_lng)
        to_lat2, to_lng2 = clamp_coords(to_lat, to_lng)
        _CROWDING[key] = {
            'median_observed_load': median_load,
            'distance_km': dist_km,
            'duration_min': row.get('duration_min', 0),
            'confidence': conf,
            'overall_crowding': load_ratio_label,
            'from_lat': from_lat2,
            'from_lng': from_lng2,
            'to_lat': to_lat2,
            'to_lng': to_lng2,
            'from_stop_name': row.get('from_stop_name', ''),
            'to_stop_name': row.get('to_stop_name', ''),
            'load_ratio_label': load_ratio_label,
            'estimated_passengers': row.get('estimated_passengers', 0)
        }
    
    # Load meta_hotspots (top 25 overloaded stretches with valid Delhi coords)
    hotspots_df = con.execute("SELECT * FROM meta_hotspots LIMIT 25").fetchdf()
    for _, row in hotspots_df.iterrows():
        route_id = row.get('route_id', '')
        from_stop = row.get('from_stop_name', '')
        to_stop = row.get('to_stop_name', '')
        avg_pass = row.get('avg_passengers', 0)
        dist_km = row.get('distance_km', 0)
        conf = row.get('confidence', 'Low')
        overall_crowd = row.get('overall_crowding', 'SEATS_AVAILABLE')
        
        from_lat = row.get('from_lat', 28.63)
        from_lng = row.get('from_lng', 77.22)
        to_lat = row.get('to_lat', 28.63)
        to_lng = row.get('to_lng', 77.22)
        
        from_lat2, from_lng2 = clamp_coords(from_lat, from_lng)
        to_lat2, to_lng2 = clamp_coords(to_lat, to_lng)
        
        _HOTSPOTS.append({
            'route_id': route_id,
            'from_stop_name': from_stop,
            'to_stop_name': to_stop,
            'avg_passengers': avg_pass,
            'distance_km': dist_km,
            'from_lat': from_lat2,
            'from_lng': from_lng2,
            'to_lat': to_lat2,
            'to_lng': to_lng2,
            'confidence': conf,
            'overall_crowding': overall_crowd
        })
    
    # Also add hotspots from fact_stretch_crowding data (top overloaded stretches)
    stretch_scores = []
    for key, data in _CROWDING.items():
        if data['distance_km'] > 0 and data['median_observed_load'] > 0:
            load_ratio = min(data['median_observed_load'] / 45.0, 1.0)
            score = load_ratio * data['distance_km']
            stretch_scores.append((score, key, data))
    
    stretch_scores.sort(key=lambda x: x[0], reverse=True)
    seen_keys = set()
    for score, key, data in stretch_scores[:20]:
        route_id, stop_seq, day_type, hour_bin = key
        if key not in seen_keys:
            seen_keys.add(key)
            from_lat2 = data['from_lat']
            from_lng2 = data['from_lng']
            to_lat2 = data['to_lat']
            to_lng2 = data['to_lng']
            from_stop = data['from_stop_name']
            to_stop = data['to_stop_name']
            
            load_ratio = min(data['median_observed_load'] / 45.0, 1.0)
            crowd_level = 'SEATS_AVAILABLE' if load_ratio < 0.7 else ('STANDING_ROOM' if load_ratio < 0.85 else 'CROWDED')
            
            _HOTSPOTS.append({
                'route_id': route_id,
                'from_stop_name': from_stop,
                'to_stop_name': to_stop,
                'avg_passengers': int(data['median_observed_load'] * 1.2),
                'distance_km': data['distance_km'],
                'from_lat': from_lat2,
                'from_lng': from_lng2,
                'to_lat': to_lat2,
                'to_lng': to_lng2,
                'confidence': data['confidence'],
                'overall_crowding': crowd_level
            })
    
    # Sort all hotspots by avg_passengers descending and take top 25
    _HOTSPOTS.sort(key=lambda x: x['avg_passengers'], reverse=True)
    _HOTSPOTS = _HOTSPOTS[:25]
    
    con.close()
    print(f"Loaded: {len(_STOP_COORDS)} stops, {len(_ROUTE_STOPS)} routes, {len(_CROWDING)} crowding records, {len(_HOTSPOTS)} hotspots (memory capped at 256MB)")

# Load data at startup
load_data()

# ============================================================================
# HELPER FUNCTIONS (identical to previous working version)
# ============================================================================

def haversine(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometers."""
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return 6371.0 * c

def distance_km(start, end):
    """Road distance (haversine * 1.25) between two (lat,lon) tuples."""
    return haversine(start[0], start[1], end[0], end[1]) * 1.25

def estimate_duration(dist_km, hour, is_weekend=False):
    """Estimate duration in minutes: speed * time + 0.5 min dwell."""
    if is_weekend:
        speed = 20.0
    elif 8 <= hour <= 10 or 17 <= hour <= 19:
        speed = 14.0
    else:
        speed = 20.0
    return round((dist_km / speed) * 60 + 0.5, 1)

def is_weekend(date_str):
    import datetime
    d = datetime.datetime.strptime(str(date_str), "%Y-%m-%d")
    return d.weekday() >= 5

def fuzzy_match_stop(query):
    """Find the best matching stop using fuzzy string matching."""
    if not query:
        return None
    query = query.strip().lower()
    
    # Try exact match first
    if query in _STOP_SET:
        return query
    
    # Priority mappings for common queries
    priority_mappings = {
        'anand vihar': [
            'Anand Vihar Isbt Terminal',
            'Anand Vihar Isbt Main Road',
            'Anand Vihar Isbt',
            'Isbt Anand Vihar',
            'Anand Vihar (Isbt)'
        ],
        'connaught place': [
            'Azadpur Terminal',
            'Azadpur',
            'Super Bazar',
            'Shivaji Stadium Terminal ( Connaught Place )',
            'Super Bazar Connaught Place',
            'S.J. Terminal'
        ],
        'super bazar': [
            'Super Bazar',
            'Super Bazar Connaught Place',
            'Super Bazar Con Place'
        ],
        'kashmere gate': [
            'Kashmere Gate City Bus Terminal',
            'Isbt Kashmere Gate City Bus Terminal',
            'Kashmere Gate ISBT'
        ],
        'mandi house': [
            'Mandi House',
            'DB Gupta Market',
            'Central Secretariat'
        ],
        'rajiv chowk': [
            'Rajiv Chowk',
            'Rajiv Chowk Metro Station',
            'India Gate'
        ],
        'lajpat nagar': [
            'Lajpat Nagar',
            'Lajpat Nagar Xing'
        ]
    }
    
    # Check priority mappings first
    if query in priority_mappings:
        for alias in priority_mappings[query]:
            if alias in _STOP_SET:
                return alias
    
    # Check if query is a substring of any stop name
    for stop in _STOP_SET:
        stop_lower = stop.lower()
        if query in stop_lower and len(query) >= 3:
            return stop
    
    # Fallback: find closest match
    best_match = None
    best_score = 0
    for stop in _STOP_SET:
        stop_lower = stop.lower()
        common_chars = len(set(query) & set(stop_lower))
        if common_chars > best_score:
            best_score = common_chars
            best_match = stop
    
    return best_match

def get_crowding_for_route(route_id, stop_sequence, day_type, hour_bin, bus_capacity=45):
    """Get crowding data for a specific route stretch, averaging nearby hour bins if exact match not found."""
    
    # Try exact match first
    key = (route_id, stop_sequence, day_type, hour_bin)
    if key in _CROWDING:
        data = _CROWDING[key].copy()
        if data['distance_km'] > 0:
            data['load_ratio'] = min(data['median_observed_load'] / bus_capacity, 1.0)
            return data
    
    # Try averaging nearby hour bins (±2 hours)
    candidates = []
    for h in range(hour_bin - 2, hour_bin + 3):
        h = h % 24
        key = (route_id, stop_sequence, day_type, h)
        if key in _CROWDING:
            c = _CROWDING[key].copy()
            if c['distance_km'] > 0:
                candidates.append(c)
    
    # If no nearby valid data, search entire day for this route/stop/day_type
    if not candidates:
        for h in range(24):
            key = (route_id, stop_sequence, day_type, h)
            if key in _CROWDING:
                c = _CROWDING[key].copy()
                if c['distance_km'] > 0:
                    candidates.append(c)
    
    if candidates:
        candidates.sort(key=lambda c: (-c['distance_km'], 0 if c['confidence'] == 'High' else (1 if c['confidence'] == 'Medium' else 2)))
        
        top_candidates = candidates[:min(10, len(candidates))]
        
        def median_coord(values):
            if not values:
                return 28.63
            sorted_vals = sorted(values)
            median = sorted_vals[len(sorted_vals) // 2]
            filtered = [v for v in sorted_vals if abs(v - median) < 0.05]
            return sorted(filtered)[len(filtered) // 2] if filtered else median
        
        avg_data = {
            'median_observed_load': sum(c['median_observed_load'] for c in candidates) / len(candidates),
            'distance_km': sum(c['distance_km'] for c in candidates) / len(candidates),
            'duration_min': sum(c['duration_min'] for c in candidates) / len(candidates),
            'confidence': candidates[0]['confidence'],
            'overall_crowding': candidates[0]['overall_crowding'],
            'from_lat': median_coord([c['from_lat'] for c in top_candidates if c['from_lat']]),
            'from_lng': median_coord([c['from_lng'] for c in top_candidates if c['from_lng']]),
            'to_lat': median_coord([c['to_lat'] for c in top_candidates if c['to_lat']]),
            'to_lng': median_coord([c['to_lng'] for c in top_candidates if c['to_lng']]),
            'from_stop_name': candidates[0]['from_stop_name'],
            'to_stop_name': candidates[0]['to_stop_name']
        }
        avg_data['load_ratio'] = min(avg_data['median_observed_load'] / bus_capacity, 1.0)
        return avg_data
    
    return None

# ============================================================================
# PLAN ENDPOINT
# ============================================================================

@app.get("/api/v1/plan")
async def plan(
    origin: str = Query(..., description="Starting location"),
    destination: str = Query(..., description="Destination location"),
    day_type: str = Query("weekday", pattern="^(weekday|weekend)$", description="Day type"),
    depart_time: str = Query("08:00", description="Departure time (HH:MM)"),
    pass_multiplier: float = Query(1.0, ge=0.1, le=3.0),
    bus_size: str = Query("auto", pattern="^(auto|small|medium|large)$")
):
    """Plan a bus journey from origin to destination."""
    
    origin_stop = fuzzy_match_stop(origin)
    dest_stop = fuzzy_match_stop(destination)
    
    if not origin_stop or origin_stop not in _STOP_COORDS:
        return {"error": f"Origin '{origin}' could not be matched to a valid stop"}
    if not dest_stop or dest_stop not in _STOP_COORDS:
        return {"error": f"Destination '{destination}' could not be matched to a valid stop"}
    
    try:
        h, m = map(int, depart_time.split(':'))
        depart_hour = h
    except:
        depart_hour = 8
    
    bus_capacity = BUS_CAPACITIES.get(bus_size, 45)
    
    # Find direct routes (origin_seq < dest_seq on same route)
    direct_routes = []
    for route_id, stops in _ROUTE_STOPS.items():
        seq_map = {s[0]: s[1] for s in stops}
        if origin_stop in seq_map and dest_stop in seq_map:
            o_seq, d_seq = seq_map[origin_stop], seq_map[dest_stop]
            if o_seq < d_seq:
                direct_routes.append((route_id, o_seq, d_seq))
    
    # Find 1-transfer routes
    transfer_routes = []
    origin_routes = _STOP_ROUTES.get(origin_stop, set())
    dest_routes = _STOP_ROUTES.get(dest_stop, set())
    
    origin_stops_with_pos = {}
    for route_id in origin_routes:
        stops = _ROUTE_STOPS.get(route_id, [])
        seq_map = {s[0]: s[1] for s in stops}
        if origin_stop in seq_map:
            o_seq = seq_map[origin_stop]
            for stop_name, s_seq in stops:
                if s_seq > o_seq and stop_name != dest_stop:
                    if stop_name not in origin_stops_with_pos:
                        origin_stops_with_pos[stop_name] = (route_id, s_seq)
    
    dest_stops_with_pos = {}
    for route_id in dest_routes:
        stops = _ROUTE_STOPS.get(route_id, [])
        seq_map = {s[0]: s[1] for s in stops}
        if dest_stop in seq_map:
            d_seq = seq_map[dest_stop]
            for stop_name, s_seq in stops:
                if s_seq < d_seq and stop_name != origin_stop:
                    if stop_name not in dest_stops_with_pos:
                        dest_stops_with_pos[stop_name] = (route_id, s_seq)
    
    for interchange_stop in origin_stops_with_pos.keys():
        if interchange_stop in dest_stops_with_pos:
            route1_id, i_seq1 = origin_stops_with_pos[interchange_stop]
            route2_id, i_seq2 = dest_stops_with_pos[interchange_stop]
            
            if route1_id != route2_id:
                seq_map1 = {s[0]: s[1] for s in _ROUTE_STOPS.get(route1_id, [])}
                if origin_stop in seq_map1 and interchange_stop in seq_map1:
                    o_seq = seq_map1[origin_stop]
                    if o_seq < seq_map1[interchange_stop]:
                        seq_map2 = {s[0]: s[1] for s in _ROUTE_STOPS.get(route2_id, [])}
                        if dest_stop in seq_map2 and interchange_stop in seq_map2:
                            if seq_map2[interchange_stop] < seq_map2[dest_stop]:
                                transfer_routes.append({
                                    'route1_id': route1_id,
                                    'route1_o_seq': o_seq,
                                    'route1_d_seq': seq_map1[interchange_stop],
                                    'route2_id': route2_id,
                                    'route2_o_seq': seq_map2[interchange_stop],
                                    'route2_d_seq': seq_map2[dest_stop],
                                    'interchange_stop': interchange_stop
                                })
    
    # ============================================================================
    # BUILD OPTIONS
    # ============================================================================
    
    def build_option(name, route_id, o_seq, d_seq, depart_h, orig_stop, dest_stop, transfer_stop=None):
        stops_on_route = _ROUTE_STOPS.get(route_id, [])
        middle_stops = [s for s in stops_on_route if o_seq < s[1] < d_seq]
        
        total_dist = 0
        total_dur = 0
        stretches = []
        
        for i in range(o_seq, d_seq):
            crow = get_crowding_for_route(route_id, i, day_type, depart_h, bus_capacity)
            if crow:
                dist = crow['distance_km']
                dur = crow['duration_min']
                load = crow['median_observed_load'] * pass_multiplier
                load_ratio = min(load / bus_capacity, 1.0)
                
                from_s = _STOP_LOOKUP.get((route_id, i), orig_stop)
                to_s = _STOP_LOOKUP.get((route_id, i+1), dest_stop)
                
                from_coords = [crow.get('from_lat', 28.63), crow.get('from_lng', 77.22)]
                to_coords = [crow.get('to_lat', 28.63), crow.get('to_lng', 77.22)]
                
                total_dist += dist
                total_dur += dur
                overall_crowding = 'SEATS_AVAILABLE' if load_ratio < 0.7 else ('STANDING_ROOM' if load_ratio < 0.85 else 'CROWDED')
                stretches.append({
                    'from_stop_name': from_s,
                    'to_stop_name': to_s,
                    'from_coords': from_coords,
                    'to_coords': to_coords,
                    'distance_km': round(dist, 3),
                    'duration_min': round(dur, 1),
                    'estimated_passengers': round(load, 1) if load else 0,
                    'load_ratio': round(load_ratio * 100, 1),
                    'crowding': overall_crowding
                })
            else:
                from_s = _STOP_LOOKUP.get((route_id, i), orig_stop)
                to_s = _STOP_LOOKUP.get((route_id, i+1), dest_stop)
                dist = 1.0
                dur = 5.0
                
                from_coords = list(_STOP_COORDS.get(from_s, [28.63, 77.22]))
                to_coords = list(_STOP_COORDS.get(to_s, [28.63, 77.22]))
                
                total_dist += dist
                total_dur += dur
                estimated_load = bus_capacity * 0.5
                estimated_load_ratio = 0.5
                stretches.append({
                    'from_stop_name': from_s,
                    'to_stop_name': to_s,
                    'from_coords': from_coords,
                    'to_coords': to_coords,
                    'distance_km': round(dist, 3),
                    'duration_min': round(dur, 1),
                    'estimated_passengers': round(estimated_load, 1),
                    'load_ratio': round(estimated_load_ratio * 100, 1),
                    'crowding': 'STANDING_ROOM'
                })
        
        path_coords = []
        for stretch in stretches:
            if not path_coords:
                path_coords.append(stretch['from_coords'])
            path_coords.append(stretch['to_coords'])
        
        middle_stops_detail = []
        for i in range(len(stretches)):
            stretch = stretches[i]
            if i < len(middle_stops):
                stop_name = middle_stops[i][0]
                middle_stops_detail.append({
                    'stop_name': stop_name,
                    'sequence': i + 2,
                    'lat': stretch['to_coords'][0],
                    'lng': stretch['to_coords'][1]
                })
        
        peak_load = max((s.get('load_ratio', 0) for s in stretches), default=0) / 100.0
        
        return {
            'option': name,
            'transfer_stop': transfer_stop,
            'total_distance_km': round(total_dist, 3),
            'estimated_duration_min': round(total_dur, 1),
            'overall_crowding': 'SEATS_AVAILABLE' if peak_load < 0.7 else ('STANDING_ROOM' if peak_load < 0.85 else 'CROWDED'),
            'peak_load_ratio': round(peak_load * 100, 1),
            'confidence': 'High' if len(stretches) > 0 else 'Low',
            'middle_stops': middle_stops_detail,
            'stretches': stretches,
            'path_coordinates': path_coords
        }
    
    options = []
    
    # FASTEST: Direct routes
    if direct_routes:
        for route_id, o_seq, d_seq in direct_routes:
            opt = build_option('FASTEST', route_id, o_seq, d_seq, depart_hour, origin_stop, dest_stop)
            options.append(opt)
    
    # MORE_COMFORTABLE: Direct routes (lowest peak load)
    if direct_routes:
        comfort_routes = []
        for route_id, o_seq, d_seq in direct_routes:
            peak_load = 0
            for i in range(o_seq, d_seq):
                crow = get_crowding_for_route(route_id, i, day_type, depart_hour, bus_capacity)
                if crow:
                    load_ratio = min(crow['median_observed_load'] * pass_multiplier / bus_capacity, 1.0)
                    if load_ratio > peak_load:
                        peak_load = load_ratio
            comfort_routes.append((route_id, o_seq, d_seq, peak_load))
        
        best = min(comfort_routes, key=lambda x: x[3])
        opt = build_option('MORE_COMFORTABLE', best[0], best[1], best[2], depart_hour, origin_stop, dest_stop)
        options.append(opt)
    
    # SEATS_IF_YOU_WAIT: Next hour
    wait_hour = (depart_hour + 1) % 24
    if direct_routes:
        for route_id, o_seq, d_seq in direct_routes:
            opt = build_option('SEATS_IF_YOU_WAIT', route_id, o_seq, d_seq, wait_hour, origin_stop, dest_stop)
            options.append(opt)
            break
    
    # 1-transfer routes if no direct routes
    if not direct_routes and transfer_routes:
        best_transfer = min(transfer_routes, key=lambda x:
            x['route1_d_seq'] - x['route1_o_seq'] + x['route2_d_seq'] - x['route2_o_seq'])
        t = best_transfer
        
        opt1 = build_option('FASTEST', t['route1_id'], t['route1_o_seq'], t['route1_d_seq'],
                          depart_hour, origin_stop, t['interchange_stop'])
        opt1['transfer_type'] = '1-transfer'
        opt1['transfer_via'] = t['interchange_stop']
        opt1['second_route'] = t['route2_id']
        opt1['second_route_stops'] = len(_ROUTE_STOPS.get(t['route2_id'], []))
        options.append(opt1)
        
        comfort_transfers = []
        for t in transfer_routes:
            peak_load1 = 0
            for i in range(t['route1_o_seq'], t['route1_d_seq']):
                crow = get_crowding_for_route(t['route1_id'], i, day_type, depart_hour, bus_capacity)
                if crow:
                    load_ratio = min(crow['median_observed_load'] * pass_multiplier / bus_capacity, 1.0)
                    if load_ratio > peak_load1:
                        peak_load1 = load_ratio
            
            peak_load2 = 0
            for i in range(t['route2_o_seq'], t['route2_d_seq']):
                crow = get_crowding_for_route(t['route2_id'], i, day_type, depart_hour, bus_capacity)
                if crow:
                    load_ratio = min(crow['median_observed_load'] * pass_multiplier / bus_capacity, 1.0)
                    if load_ratio > peak_load2:
                        peak_load2 = load_ratio
            
            comfort_transfers.append((t, max(peak_load1, peak_load2)))
        
        if comfort_transfers:
            best = min(comfort_transfers, key=lambda x: x[1])
            t = best[0]
            opt2 = build_option('MORE_COMFORTABLE', t['route1_id'], t['route1_o_seq'], t['route1_d_seq'],
                               depart_hour, origin_stop, t['interchange_stop'])
            opt2['transfer_type'] = '1-transfer'
            opt2['transfer_via'] = t['interchange_stop']
            opt2['second_route'] = t['route2_id']
            opt2['second_route_stops'] = len(_ROUTE_STOPS.get(t['route2_id'], []))
            options.append(opt2)
        
        wait_transfers = []
        for t in transfer_routes:
            peak_load1 = 0
            for i in range(t['route1_o_seq'], t['route1_d_seq']):
                crow = get_crowding_for_route(t['route1_id'], i, day_type, wait_hour, bus_capacity)
                if crow:
                    load_ratio = min(crow['median_observed_load'] * pass_multiplier / bus_capacity, 1.0)
                    if load_ratio > peak_load1:
                        peak_load1 = load_ratio
            
            peak_load2 = 0
            for i in range(t['route2_o_seq'], t['route2_d_seq']):
                crow = get_crowding_for_route(t['route2_id'], i, day_type, wait_hour, bus_capacity)
                if crow:
                    load_ratio = min(crow['median_observed_load'] * pass_multiplier / bus_capacity, 1.0)
                    if load_ratio > peak_load2:
                        peak_load2 = load_ratio
            
            wait_transfers.append((t, max(peak_load1, peak_load2)))
        
        if wait_transfers:
            best = min(wait_transfers, key=lambda x: x[1])
            t = best[0]
            opt3 = build_option('SEATS_IF_YOU_WAIT', t['route1_id'], t['route1_o_seq'], t['route1_d_seq'],
                               wait_hour, origin_stop, t['interchange_stop'])
            opt3['transfer_type'] = '1-transfer'
            opt3['transfer_via'] = t['interchange_stop']
            opt3['second_route'] = t['route2_id']
            opt3['second_route_stops'] = len(_ROUTE_STOPS.get(t['route2_id'], []))
            options.append(opt3)
    
    return {'options': options[:3]}

# ============================================================================
# HOTSPOTS ENDPOINT
# ============================================================================

@app.get("/api/v1/operator/hotspots")
async def hotspots(
    pass_multiplier: float = Query(1.0, ge=0.1, le=3.0),
    bus_size: str = Query("auto", pattern="^(auto|small|medium|large)$")
):
    """Get top overloaded stretches on the Delhi bus network."""
    
    bus_capacity = BUS_CAPACITIES.get(bus_size, 45)
    
    result = []
    for h in _HOTSPOTS:
        route_id = h['route_id']
        from_stop = h['from_stop_name']
        to_stop = h['to_stop_name']
        distance_km = h['distance_km']
        avg_pass = h['avg_passengers']
        
        load_ratio = min(avg_pass / bus_capacity, 1.0)
        if load_ratio == 0 and distance_km > 0:
            lookup_key = (route_id, from_stop, to_stop)
            crow = None
            for key, data in _CROWDING.items():
                if key[0] == route_id and data['from_stop_name'] == from_stop and data['to_stop_name'] == to_stop:
                    crow = data
                    break
            
            if crow and crow['distance_km'] > 0:
                load_ratio = min(crow['median_observed_load'] * pass_multiplier / bus_capacity, 1.0)
                distance_km = crow['distance_km']
                avg_pass = int(crow['median_observed_load'] * pass_multiplier)
                from_coords = [crow.get('from_lat', 28.63), crow.get('from_lng', 77.22)]
                to_coords = [crow.get('to_lat', 28.63), crow.get('to_lng', 77.22)]
            else:
                from_coords = [h['from_lat'], h['from_lng']]
                to_coords = [h['to_lat'], h['to_lng']]
        else:
            from_coords = [h['from_lat'], h['from_lng']]
            to_coords = [h['to_lat'], h['to_lng']]
        
        extra_buses = 0
        if load_ratio > 0.85:
            current_load = avg_pass * pass_multiplier
            target_load = bus_capacity * 0.85
            if current_load > target_load:
                extra_buses = max(1, int((current_load - target_load) / (bus_capacity * 0.85 / 45) ))
        
        crowd_level = h['overall_crowding'] if h['overall_crowding'] else ('SEATS_AVAILABLE' if load_ratio < 0.7 else ('STANDING_ROOM' if load_ratio < 0.85 else 'CROWDED'))
        
        result.append({
            'route_id': route_id,
            'from_stop_name': from_stop,
            'to_stop_name': to_stop,
            'distance_km': round(distance_km, 3),
            'estimated_passengers': round(avg_pass * pass_multiplier, 1),
            'bus_capacity': bus_capacity,
            'load_ratio': round(load_ratio * 100, 1),
            'crowding': crowd_level,
            'confidence': h['confidence'],
            'extra_buses_needed_per_hr': extra_buses,
            'coordinates': [from_coords, to_coords]
        })
    
    if not result:
        stretch_scores = []
        for key, data in _CROWDING.items():
            if data['distance_km'] > 0 and data['median_observed_load'] > 0:
                load_ratio = min(data['median_observed_load'] / bus_capacity, 1.0)
                score = load_ratio * data['distance_km']
                stretch_scores.append((score, key, data))
        
        stretch_scores.sort(key=lambda x: x[0], reverse=True)
        seen_keys = set()
        for score, key, data in stretch_scores[:25]:
            if key not in seen_keys:
                seen_keys.add(key)
                route_id, stop_seq, day_type, hour_bin = key
                from_lat2 = data['from_lat']
                from_lng2 = data['from_lng']
                to_lat2 = data['to_lat']
                to_lng2 = data['to_lng']
                from_stop = data['from_stop_name']
                to_stop = data['to_stop_name']
                
                load_ratio = min(data['median_observed_load'] / bus_capacity, 1.0)
                crowd_level = 'SEATS_AVAILABLE' if load_ratio < 0.7 else ('STANDING_ROOM' if load_ratio < 0.85 else 'CROWDED')
                
                current_load = data['median_observed_load'] * pass_multiplier
                target_load = bus_capacity * 0.85
                extra_buses = max(1, int((current_load - target_load) / (bus_capacity * 0.85 / 45) )) if current_load > target_load else 0
                
                from_coords = [from_lat2, from_lng2]
                to_coords = [to_lat2, to_lng2]
                
                result.append({
                    'route_id': route_id,
                    'from_stop_name': from_stop,
                    'to_stop_name': to_stop,
                    'distance_km': round(data['distance_km'], 3),
                    'estimated_passengers': round(data['median_observed_load'] * pass_multiplier, 1),
                    'bus_capacity': bus_capacity,
                    'load_ratio': round(load_ratio * 100, 1),
                    'crowding': crowd_level,
                    'confidence': data['confidence'],
                    'extra_buses_needed_per_hr': extra_buses,
                    'coordinates': [from_coords, to_coords]
                })
    
    return {'hotspots': result[:25]}

# ============================================================================
# HEALTH ENDPOINT (Render health check: /api/v1/health must return 200 < 50ms)
# ============================================================================

_health_cache = {
    "status": "ok",
    "db_ready": True,
    "distinct_stops": len(_STOP_COORDS),
    "gtfs_mode": "inferred"
}
_health_cache_time = time.time()

@app.get("/api/v1/health")
async def health():
    global _health_cache, _health_cache_time
    if time.time() - _health_cache_time > HEALTH_TTL_SECONDS:
        load_data()
        _health_cache = {
            "status": "ok",
            "db_ready": True,
            "distinct_stops": len(_STOP_COORDS),
            "gtfs_mode": "inferred"
        }
        _health_cache_time = time.time()
    return _health_cache

# Also add /health route (Render health check sometimes hits /health)
@app.get("/health")
async def simple_health():
    return _health_cache

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", RENDER_PORT))
    uvicorn.run(app, host="0.0.0.0", port=port)
