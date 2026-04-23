import React, { useEffect, useState } from 'react';
import { MapContainer, Polyline, CircleMarker, Circle, Popup, useMapEvents } from 'react-leaflet';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import 'leaflet/dist/leaflet.css';
import './App.css';

const MapController = ({ roads, setZoomLevel }) => {
    const map = useMapEvents({
        zoom: () => {
            setZoomLevel(map.getZoom());
        }
    });

    useEffect(() => {
        setZoomLevel(map.getZoom());
        if (roads && roads.length > 0) {
            const bounds = [];
            roads.forEach(road => {
                if (road.path) road.path.forEach(pt => bounds.push(pt));
            });
            if (bounds.length > 0) {
                map.fitBounds(bounds, { padding: [20, 20] });
            }
        }
    }, [roads, map, setZoomLevel]);
    return null;
};

const App = () => {
    const [zoomLevel, setZoomLevel] = useState(16);
    const [state, setState] = useState(null);
    const [algo, setAlgo] = useState('greedy');
    const [roads, setRoads] = useState([]);
    const [history, setHistory] = useState([]);
    
    // UI Sliders state
    const [numHubs, setNumHubs] = useState(15);
    const [numCars, setNumCars] = useState(50);
    const [eventRate, setEventRate] = useState(0.05); // percentage for easier display

    useEffect(() => {
        axios.get('http://localhost:8000/roads').then(res => {
            if (res.data) setRoads(res.data);
        }).catch(e => console.error(e));

        const interval = setInterval(() => {
            axios.get('http://localhost:8000/state').then(res => {
                setState(res.data);
                setAlgo(res.data.algorithm);
                
                // Update LineChart history with current stats
                setHistory(prev => {
                    const newHist = [...prev, { 
                        step: res.data.step, 
                        Generated: parseFloat(res.data.total_generated_mb.toFixed(2)), 
                        Offloaded: parseFloat(res.data.total_offloaded_mb.toFixed(2)),
                        Drops: Math.min(100, res.data.total_network_drops || 0) // capped for scale
                    }];
                    // If step gets reset to 0, flush history
                    if (res.data.step === 0 && prev.length > 0) return [];
                    // Keep last 60 entries for the chart window
                    return newHist.length > 60 ? newHist.slice(1) : newHist;
                });

            }).catch(e => console.error(e));
        }, 150); // Changed from 50 to 150 to reduce react-leaflet re-renders and fix abhorrent performance
        
        return () => clearInterval(interval);
    }, []);

    const toggleAlgo = () => {
        const newAlgo = algo === 'greedy' ? 'predictive' : 'greedy';
        axios.post('http://localhost:8000/algorithm', { algorithm: newAlgo }).then(() => {
            setAlgo(newAlgo);
        });
    };

    const startSim = () => axios.post('http://localhost:8000/start');
    const pauseSim = () => axios.post('http://localhost:8000/pause');
    const stepSim = () => axios.post('http://localhost:8000/step');

    const handleReset = () => {
        setHistory([]);
        axios.post('http://localhost:8000/reset', {
            num_hubs: numHubs,
            num_vehicles: numCars,
            event_rate: eventRate / 100 // convert back to base decimal for backend
        }).then(() => {
            console.log("Simulation Reset Success");
        });
    };

    const downloadReport = async () => {
        try {
            const res = await axios.get('http://localhost:8000/report');
            const blob = new Blob([JSON.stringify(res.data, null, 2)], { type: "application/json" });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `edge-sim-report-${res.data.algorithm}.json`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (e) {
            console.error("Failed to generate report", e);
            alert("Failed to generate report.");
        }
    };

    if (!state) return <div style={{ color: '#fff', padding: '20px' }}>Loading Edge Simulation Data... (This takes a few seconds)</div>;

    return (
        <div style={{ display: 'flex', height: '100vh', backgroundColor: '#121212', color: '#fff', fontFamily: 'sans-serif' }}>
            
            {/* Sidebar Dashboard */}
            <div style={{ width: '400px', display: 'flex', flexDirection: 'column', borderRight: '1px solid #333' }}>
                <div style={{ padding: '20px', background: '#1e1e1e', borderBottom: '1px solid #333' }}>
                    <h2 style={{ margin: '0 0 15px 0', fontSize: '18px' }}>Edge-to-Cloud Sim</h2>
                    <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
                        <button onClick={startSim} style={btnStyle}>Start</button>
                        <button onClick={pauseSim} style={btnStyle}>Pause</button>
                        <button onClick={stepSim} style={btnStyle}>Step</button>
                        <button onClick={handleReset} style={{ ...btnStyle, background: '#ba3030' }}>Reset</button>
                    </div>
                    <button onClick={toggleAlgo} style={{ ...btnStyle, width: '100%', background: '#007acc', marginBottom: '10px' }}>
                        Algorithm: {algo.toUpperCase()}
                    </button>
                    <button onClick={downloadReport} style={{ ...btnStyle, width: '100%', background: '#4CAF50', marginBottom: '15px' }}>
                        Download Performance Report
                    </button>
                    
                    <div style={{ fontSize: '12px', marginTop: '10px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                            <span>Vehicles ({numCars})</span>
                            <input type="range" min="10" max="250" value={numCars} onChange={e => setNumCars(parseInt(e.target.value))} style={{ width: '150px' }}/>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                            <span>Hubs ({numHubs})</span>
                            <input type="range" min="2" max="50" value={numHubs} onChange={e => setNumHubs(parseInt(e.target.value))} style={{ width: '150px' }}/>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span>Event % ({eventRate}%)</span>
                            <input type="range" min="0.01" max="2.0" step="0.01" value={eventRate} onChange={e => setEventRate(parseFloat(e.target.value))} style={{ width: '150px' }}/>
                        </div>
                    </div>
                </div>

                <div style={{ padding: '20px', borderBottom: '1px solid #333' }}>
                    <div style={{ marginBottom: '10px' }}><strong>Step:</strong> {state.step}</div>
                    <div style={{ marginBottom: '10px', color: '#ff7373' }}><strong>Generated:</strong> {state.total_generated_mb.toFixed(2)} MB</div>
                    <div style={{ marginBottom: '10px', color: '#00ff00' }}><strong>Offloaded:</strong> {state.total_offloaded_mb.toFixed(2)} MB</div>
                    <div style={{ marginBottom: '20px', color: '#ffa500' }}><strong>Network Drops:</strong> {state.total_network_drops || 0} packets</div>

                    <div style={{ height: '200px', width: '100%' }}>
                        <ResponsiveContainer>
                            <LineChart data={history} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                <XAxis dataKey="step" stroke="#888" tick={{fontSize: 10}} />
                                <YAxis stroke="#888" tick={{fontSize: 10}} />
                                <Tooltip contentStyle={{ backgroundColor: '#222', border: 'none' }} itemStyle={{ color: '#fff' }} />
                                <Line type="monotone" dataKey="Generated" stroke="#ff7373" dot={false} isAnimationActive={false} strokeWidth={2} />
                                <Line type="monotone" dataKey="Offloaded" stroke="#00ff00" dot={false} isAnimationActive={false} strokeWidth={2} />
                                <Line type="monotone" dataKey="Drops" stroke="#ffa500" dot={false} isAnimationActive={false} strokeWidth={1} strokeDasharray="3 3"/>
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div style={{ padding: '15px 20px', borderBottom: '1px solid #333' }}>
                    <h3 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#888', textTransform: 'uppercase' }}>Map Legend</h3>
                    
                    <div style={{ display: 'flex', alignItems: 'center', marginBottom: '6px', fontSize: '12px' }}>
                        <span style={{ display: 'inline-block', width: '12px', height: '12px', borderRadius: '50%', background: '#ff3333', marginRight: '6px' }}></span>
                        Standard Vehicle
                    </div>
                    
                    <div style={{ display: 'flex', alignItems: 'center', marginBottom: '6px', fontSize: '12px' }}>
                        <span style={{ display: 'inline-block', width: '12px', height: '12px', borderRadius: '50%', background: '#ffaa00', marginRight: '6px', boxShadow: '0 0 5px #ffaa00' }}></span>
                        Vehicle (Carrying Event Data)
                    </div>
                    
                    <div style={{ display: 'flex', alignItems: 'center', marginBottom: '6px', fontSize: '12px' }}>
                        <span style={{ display: 'inline-block', width: '14px', height: '14px', borderRadius: '50%', border: '2px solid #00aa00', background: 'rgba(0, 170, 0, 0.1)', marginRight: '6px' }}></span>
                        Idle WiFi Edge Hub
                    </div>
                    
                    <div style={{ display: 'flex', alignItems: 'center', marginBottom: '6px', fontSize: '12px' }}>
                        <span style={{ display: 'inline-block', width: '14px', height: '14px', borderRadius: '50%', border: '2px solid #00ffff', background: 'rgba(0, 255, 255, 0.3)', marginRight: '6px', boxShadow: '0 0 8px #00ffff' }}></span>
                        Transmitting Edge Hub
                    </div>
                    
                    <div style={{ display: 'flex', alignItems: 'center', marginBottom: '6px', fontSize: '12px' }}>
                        <span style={{ display: 'inline-block', width: '14px', height: '14px', borderRadius: '50%', border: '2px solid #555', background: 'rgba(85, 85, 85, 0.1)', marginRight: '6px' }}></span>
                        Offline / Dead Edge Hub
                    </div>
                </div>

                <div style={{ flex: 1, padding: '20px', overflowY: 'auto' }}>
                    <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Live Event Feed</h3>
                    {state.recent_events && state.recent_events.map((evt, i) => (
                        <div key={i} style={{ fontSize: '11px', padding: '5px 0', borderBottom: '1px solid #222', color: '#ccc' }}>
                            {evt}
                        </div>
                    ))}
                    {(!state.recent_events || state.recent_events.length === 0) && (
                        <div style={{ fontSize: '12px', color: '#666' }}>No events offloaded yet...</div>
                    )}
                </div>
            </div>
            
            {/* Main Map View */}
            <div style={{ flex: 1, position: 'relative' }}>
                <MapContainer 
                    center={[33.6515, 73.0801]} 
                    zoom={16} 
                    style={{ height: '100%', width: '100%', background: '#080808' }}
                    zoomControl={true}
                    dragging={true}
                    scrollWheelZoom={true}
                    doubleClickZoom={true}
                >
                    <MapController roads={roads} setZoomLevel={setZoomLevel} />
                    
                    {roads.map((road, idx) => {
                        // AnyLogic style: simple, clean, non-exponential schematic widths
                        const laneScale = zoomLevel >= 16 ? 4 : (zoomLevel >= 14 ? 2 : 1);
                        return (
                            <React.Fragment key={`road-grp-${idx}`}>
                                <Polyline 
                                    positions={road.path} 
                                    pathOptions={{ color: '#4a4a4a', weight: road.lanes * laneScale }} 
                                />
                                {road.lanes > 1 && zoomLevel >= 14 && (
                                    <Polyline 
                                        positions={road.path} 
                                        pathOptions={{ color: '#ffffff', weight: Math.max(1, laneScale * 0.25), dashArray: "5 10", opacity: 0.5 }} 
                                    />
                                )}
                            </React.Fragment>
                        );
                    })}
                    
                    {state.hubs && state.hubs.map((hub, idx) => {
                        const hubColor = !hub.online ? '#555555' : (hub.active ? '#00ffff' : '#00aa00');
                        return (
                        <Circle 
                            key={`hub-${idx}`}
                            center={[hub.lat, hub.lon]} 
                            radius={hub.range} 
                            pathOptions={{ 
                                color: hubColor, 
                                fillColor: hubColor, 
                                fillOpacity: !hub.online ? 0.1 : (hub.active ? 0.3 : 0.05), 
                                weight: hub.active && hub.online ? 3 : 1 
                            }}
                        >
                            <Popup>WiFi Hub {hub.id} <br/> Rate: {hub.rate} MB/s <br/> Status: {!hub.online ? 'OFFLINE' : (hub.active ? 'Transmitting' : 'Idle')}</Popup>
                        </Circle>
                    )})}

                    {state.vehicles && state.vehicles.map((v, idx) => {
                        const hasEvents = v.events && v.events.length > 0;
                        
                        // Find active hub connection for the network visual
                        const hubConn = v.current_bw_alloc > 0 ? state.hubs.find(h => {
                            // Find closest active hub
                            const dist = Math.sqrt(Math.pow(h.lat - v.lat, 2) + Math.pow(h.lon - v.lon, 2));
                            return h.active && h.online && dist <= (h.range * 0.000009); // Loose approx check
                        }) : null;

                        return (
                            <React.Fragment key={`vgrp-${idx}`}>
                                {hubConn && (
                                    <Polyline 
                                        positions={[[v.lat, v.lon], [hubConn.lat, hubConn.lon]]} 
                                        pathOptions={{ color: v.total_packet_drops > 0 ? '#ff0000' : '#00ffff', weight: 2, dashArray: "2 6", opacity: 0.8 }} 
                                    />
                                )}
                                <CircleMarker 
                                    center={[v.lat, v.lon]}
                                    radius={hasEvents ? 5.5 : 3.5}
                                    pathOptions={{ 
                                        color: hasEvents ? '#ffaa00' : '#ff3333', 
                                        fillColor: hasEvents ? '#ffaa00' : '#ff3333', 
                                        fillOpacity: 1, 
                                        weight: 1 
                                    }}
                                >
                                    <Popup>
                                        Vehicle {v.id}<br/>
                                        Buffer: {v.buffer_mb.toFixed(2)} MB<br/>
                                        Speed: {v.speed ? v.speed.toFixed(6) : 0} deg/s<br/>
                                        Allocated BW: {v.current_bw_alloc ? v.current_bw_alloc.toFixed(3) : 0} MB/tick<br/>
                                        Pkt Drops: {v.total_packet_drops || 0} events<br/>
                                        Events: {hasEvents ? v.events.map(ev => ev.type).join(', ') : 'None'}
                                    </Popup>
                                </CircleMarker>
                            </React.Fragment>
                        );
                    })}
                    {state.signals && state.signals.map((sig, idx) => (
                        <CircleMarker
                            key={`sig-${idx}`}
                            center={[sig.lat, sig.lon]}
                            radius={4}
                            pathOptions={{
                                color: sig.state === 'RED' ? '#ff3333' : '#33ff33',
                                fillColor: sig.state === 'RED' ? '#ff3333' : '#33ff33',
                                fillOpacity: 0.9,
                                weight: 1
                            }}
                        >
                            <Popup>Traffic Light: {sig.state}</Popup>
                        </CircleMarker>
                    ))}
                </MapContainer>
            </div>

            {/* Right Sidebar - UI Congestion Indicators */}
            <div style={{ width: '320px', display: 'flex', flexDirection: 'column', borderLeft: '1px solid #333', background: '#1e1e1e', padding: '20px', overflowY: 'auto' }}>
                <h2 style={{ margin: '0 0 15px 0', fontSize: '18px', color: '#ff9900' }}>Live Congestion Monitors</h2>
                
                <h3 style={{ fontSize: '14px', borderBottom: '1px solid #444', paddingBottom: '5px' }}>Top Network Bottlenecks</h3>
                {state.hubs && [...state.hubs].sort((a,b) => b.rate - a.rate).slice(0, 5).map((hub, idx) => (
                    <div key={`hc-${idx}`} style={{ marginBottom: '10px', fontSize: '12px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                            <span>Hub {hub.id} {!hub.online ? '(Offline)' : ''}</span>
                            <span style={{ color: hub.active ? '#00ff00' : '#888' }}>{hub.active ? 'Active' : 'Idle'}</span>
                        </div>
                        <div style={{ background: '#333', height: '6px', borderRadius: '3px', width: '100%', overflow:'hidden' }}>
                            <div style={{ background: !hub.online ? '#555' : (hub.active ? '#00ffff' : '#444'), height: '100%', width: `${Math.min(100, hub.active ? 100 : 0)}%` }}></div>
                        </div>
                    </div>
                ))}

                <h3 style={{ fontSize: '14px', borderBottom: '1px solid #444', paddingBottom: '5px', marginTop: '20px' }}>Top Jammed Vehicles</h3>
                {state.vehicles && [...state.vehicles].sort((a,b) => b.total_packet_drops - a.total_packet_drops).slice(0, 5).map((v, idx) => (
                    <div key={`vc-${idx}`} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid #2a2a2a', fontSize: '12px' }}>
                        <span>Veh {v.id}</span>
                        <span style={{ color: v.total_packet_drops > 10 ? '#ff3333' : '#ffaa00' }}>{v.total_packet_drops || 0} drops</span>
                    </div>
                ))}
                
                <h3 style={{ fontSize: '14px', borderBottom: '1px solid #444', paddingBottom: '5px', marginTop: '20px' }}>Priority Buf Risks</h3>
                {state.vehicles && [...state.vehicles].sort((a,b) => b.buffer_mb - a.buffer_mb).slice(0, 5).map((v, idx) => (
                    <div key={`vb-${idx}`} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid #2a2a2a', fontSize: '12px' }}>
                        <span>Veh {v.id}</span>
                        <span style={{ color: v.buffer_mb > 4 ? '#ff3333' : '#aaaaaa' }}>{v.buffer_mb.toFixed(2)} MB</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

const btnStyle = {
    flex: 1,
    padding: '8px',
    background: '#444',
    border: 'none',
    color: '#fff',
    cursor: 'pointer',
    borderRadius: '4px',
    fontWeight: 'bold',
    textTransform: 'uppercase',
    fontSize: '12px'
};

export default App;
