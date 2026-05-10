import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import json

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="التركيب الذري | Atomic Structure",
    layout="wide",
    page_icon="⚛️",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: #1a2236;
    --accent-cyan: #00e5ff;
    --accent-orange: #ff6b35;
    --accent-purple: #a855f7;
    --accent-green: #22d3ee;
    --text-primary: #f0f4ff;
    --text-secondary: #94a3b8;
    --border-color: #1e293b;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #0a0e1a 100%);
    color: var(--text-primary);
}

.main > div { padding-top: 1rem; }

h1, h2, h3, h4 {
    font-family: 'Tajawal', sans-serif !important;
    color: var(--text-primary) !important;
}

.stMarkdown, p, span, div, label {
    font-family: 'Tajawal', sans-serif !important;
}

.card {
    background: linear-gradient(145deg, rgba(26,34,54,0.95), rgba(17,24,39,0.9));
    border: 1px solid rgba(0,229,255,0.15);
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 30px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
}

.card-accent {
    border-color: rgba(0,229,255,0.3);
    box-shadow: 0 0 20px rgba(0,229,255,0.3), 0 4px 30px rgba(0,0,0,0.3);
}

.equation-box {
    background: rgba(0,229,255,0.05);
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 12px;
    padding: 16px 24px;
    margin: 12px 0;
    direction: ltr;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2em;
    color: var(--accent-cyan);
}

.example-box {
    background: linear-gradient(145deg, rgba(255,107,53,0.08), rgba(255,107,53,0.03));
    border-left: 4px solid var(--accent-orange);
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 12px 0;
}

.info-box {
    background: linear-gradient(145deg, rgba(168,85,247,0.08), rgba(168,85,247,0.03));
    border-left: 4px solid var(--accent-purple);
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 12px 0;
}

.section-title {
    font-size: 1.8em;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(0,229,255,0.2);
    margin-bottom: 24px;
}

.sub-title {
    font-size: 1.3em;
    font-weight: 700;
    color: var(--accent-cyan);
    margin: 20px 0 12px 0;
}

.sidebar-section {
    background: rgba(26,34,54,0.8);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    border: 1px solid rgba(0,229,255,0.1);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #0a0e1a 100%);
}
[data-testid="stSidebar"] * {
    font-family: 'Tajawal', sans-serif !important;
}

.author-badge {
    background: linear-gradient(135deg, rgba(0,229,255,0.1), rgba(168,85,247,0.1));
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 12px;
    padding: 12px 16px;
    text-align: center;
    margin-top: 20px;
}
.author-badge p {
    color: var(--accent-cyan);
    font-weight: 700;
    font-size: 0.95em;
    margin: 0;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,229,255,0.3), transparent);
    margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
H_PLANCK = 6.63e-34
H_BAR = 1.055e-34
C_LIGHT = 3e8
ME = 9.11e-31
EV_TO_J = 1.6e-19
R_H = 1.097e7

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def calc_energy_level(n):
    return -13.6 / (n ** 2)

def calc_photon_energy(ni, nf):
    return abs(calc_energy_level(nf) - calc_energy_level(ni))

def calc_wavelength(energy_ev):
    energy_j = energy_ev * EV_TO_J
    if energy_j == 0:
        return float('inf')
    return H_PLANCK * C_LIGHT / energy_j

def calc_frequency(energy_ev):
    energy_j = energy_ev * EV_TO_J
    if energy_j == 0:
        return 0
    return energy_j / H_PLANCK

def calc_angular_momentum(n):
    return n * H_BAR

def calc_rydberg_wavelength(ni, nf):
    return 1.0 / (R_H * abs(1.0/nf**2 - 1.0/ni**2))

def calc_de_broglie_wavelength(mass, velocity):
    if mass == 0 or velocity == 0:
        return float('inf')
    return H_PLANCK / (mass * velocity)

def wavelength_to_rgb(wl_nm):
    if wl_nm < 380:
        return '#8800ff'
    elif wl_nm < 440:
        r = -(wl_nm - 440) / (440 - 380)
        return f'rgb({int(r*255)},0,255)'
    elif wl_nm < 490:
        g = (wl_nm - 440) / (490 - 440)
        return f'rgb(0,{int(g*255)},255)'
    elif wl_nm < 510:
        b = -(wl_nm - 510) / (510 - 490)
        return f'rgb(0,255,{int(b*255)})'
    elif wl_nm < 580:
        r = (wl_nm - 510) / (580 - 510)
        return f'rgb({int(r*255)},255,0)'
    elif wl_nm < 645:
        g = -(wl_nm - 645) / (645 - 580)
        return f'rgb(255,{int(g*255)},0)'
    elif wl_nm <= 780:
        return 'rgb(255,0,0)'
    else:
        return '#440000'

# ============================================================
# 3D BOHR ATOM (no f-string to avoid JS brace conflicts)
# ============================================================
def bohr_atom_3d(current_n=1, transition_to=None, emit=False):
    orbit_radii = [1.0, 2.0, 3.0, 3.8, 4.5]
    orbit_colors = ['#ff3366', '#00ff88', '#3388ff', '#ffcc00', '#cc66ff']
    energy_vals = [-13.6, -3.4, -1.51, -0.85, -0.544]

    if transition_to is not None and transition_to != current_n:
        target_r = orbit_radii[min(transition_to, 4) - 1]
        trans_code = "let transitionActive=true;let transStartR="+str(orbit_radii[min(current_n,4)-1])+";let transEndR="+str(target_r)+";let transProgress=0;let transSpeed=0.015;"
    else:
        trans_code = "let transitionActive=false;"

    photon_code = ""
    if emit and transition_to is not None and transition_to < current_n:
        photon_code = """let photonActive=true;let photonAngle=0;let photonDist=0;
        let photonColor=new THREE.Color().setHSL(0.08,1.0,0.7);
        let photonMesh=new THREE.Mesh(new THREE.SphereGeometry(0.08,8,8),new THREE.MeshBasicMaterial({color:photonColor,transparent:true,opacity:0.9}));
        scene.add(photonMesh);let photonTrail=[];
        for(let i=0;i<30;i++){let t=new THREE.Mesh(new THREE.SphereGeometry(0.04,4,4),new THREE.MeshBasicMaterial({color:photonColor,transparent:true,opacity:0.4*(1-i/30)}));scene.add(t);photonTrail.push(t);}"""

    sound_code = ""
    if transition_to is not None and transition_to != current_n:
        de = abs(energy_vals[min(transition_to,4)-1] - energy_vals[min(current_n,4)-1])
        freq_snd = max(200, min(2000, de * 150))
        sound_code = "try{let a=new(window.AudioContext||window.webkitAudioContext)();let o=a.createOscillator();let g=a.createGain();o.connect(g);g.connect(a.destination);o.frequency.value="+str(freq_snd)+";o.type='sine';g.gain.setValueAtTime(0.15,a.currentTime);g.gain.exponentialRampToValueAtTime(0.001,a.currentTime+0.8);o.start(a.currentTime);o.stop(a.currentTime+0.8);}catch(e){}"

    html = """<div id="bohr-container" style="width:100%;height:500px;border-radius:16px;overflow:hidden;border:1px solid rgba(0,229,255,0.15);background:#060a14;"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
(function(){
var container=document.getElementById('bohr-container');
var w=container.clientWidth,h=container.clientHeight;
var scene=new THREE.Scene();
var camera=new THREE.PerspectiveCamera(55,w/h,0.1,1000);
camera.position.set(5,4,6);
var renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
renderer.setSize(w,h);renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
renderer.setClearColor(0x060a14);container.appendChild(renderer.domElement);
var controls=new THREE.OrbitControls(camera,renderer.domElement);
controls.enableDamping=true;controls.dampingFactor=0.05;controls.autoRotate=true;controls.autoRotateSpeed=0.5;
scene.add(new THREE.AmbientLight(0x334466,0.6));
var pointLight=new THREE.PointLight(0xff6644,2,50);pointLight.position.set(0,0,0);scene.add(pointLight);
var dirLight=new THREE.DirectionalLight(0x4488ff,0.5);dirLight.position.set(5,5,5);scene.add(dirLight);
var nucleus=new THREE.Mesh(new THREE.SphereGeometry(0.25,32,32),new THREE.MeshPhongMaterial({color:0xff4400,emissive:0xff2200,emissiveIntensity:0.6}));
scene.add(nucleus);
scene.add(new THREE.Mesh(new THREE.SphereGeometry(0.35,16,16),new THREE.MeshBasicMaterial({color:0xff6600,transparent:true,opacity:0.15})));
var radii=__RADII__;var colors=__COLORS__;
for(var i=0;i<5;i++){var o=new THREE.Mesh(new THREE.TorusGeometry(radii[i],0.015,8,128),new THREE.MeshBasicMaterial({color:colors[i],transparent:true,opacity:0.35}));o.rotation.x=Math.PI/2;scene.add(o);}
var currentR=radii[__CN__];
var electron=new THREE.Mesh(new THREE.SphereGeometry(0.1,16,16),new THREE.MeshPhongMaterial({color:0x00ffff,emissive:0x00aaff,emissiveIntensity:0.8}));
scene.add(electron);
var eGlow=new THREE.Mesh(new THREE.SphereGeometry(0.18,8,8),new THREE.MeshBasicMaterial({color:0x00ffff,transparent:true,opacity:0.2}));
scene.add(eGlow);
var trailParts=[];for(var i=0;i<20;i++){var tp=new THREE.Mesh(new THREE.SphereGeometry(0.03,4,4),new THREE.MeshBasicMaterial({color:0x00ffff,transparent:true,opacity:0.3*(1-i/20)}));scene.add(tp);trailParts.push(tp);}
var trailPositions=[];
var starGeo=new THREE.BufferGeometry();var starPos=[];
for(var i=0;i<500;i++){starPos.push((Math.random()-0.5)*100,(Math.random()-0.5)*100,(Math.random()-0.5)*100);}
starGeo.setAttribute('position',new THREE.Float32BufferAttribute(starPos,3));
scene.add(new THREE.Points(starGeo,new THREE.PointsMaterial({color:0xffffff,size:0.1,transparent:true,opacity:0.6})));
function makeLabel(text,pos,color){
var canvas=document.createElement('canvas');canvas.width=256;canvas.height=64;
var ctx=canvas.getContext('2d');ctx.font='bold 24px monospace';ctx.fillStyle=color||'#00e5ff';ctx.textAlign='center';ctx.fillText(text,128,40);
var tex=new THREE.CanvasTexture(canvas);var mat=new THREE.SpriteMaterial({map:tex,transparent:true,opacity:0.8});
var sprite=new THREE.Sprite(mat);sprite.position.copy(pos);sprite.scale.set(1.5,0.4,1);scene.add(sprite);
}
var energies=__ENERGIES__;
for(var i=0;i<5;i++){makeLabel('n='+(i+1)+' E='+energies[i].toFixed(2)+' eV',new THREE.Vector3(radii[i]+0.5,0.3,0),colors[i]);}
var angle=0;var speed=0.02;
__TRANS_CODE__
__PHOTON_CODE__
__SOUND_CODE__
function animate(){
requestAnimationFrame(animate);angle+=speed;
if(transitionActive){transProgress+=transSpeed;if(transProgress>=1){transProgress=1;transitionActive=false;}var t=transProgress<0.5?2*transProgress*transProgress:1-Math.pow(-2*transProgress+2,2)/2;currentR=transStartR+(transEndR-transStartR)*t;}
var ex=currentR*Math.cos(angle);var ez=currentR*Math.sin(angle);
electron.position.set(ex,0,ez);eGlow.position.set(ex,0,ez);
trailPositions.unshift({x:ex,z:ez});if(trailPositions.length>20)trailPositions.pop();
for(var i=0;i<trailParts.length;i++){if(i<trailPositions.length){trailParts[i].position.set(trailPositions[i].x,0,trailPositions[i].z);}}
if(typeof photonActive!=='undefined'&&photonActive){photonAngle+=0.03;photonDist+=0.08;
photonMesh.position.set(photonDist*Math.cos(photonAngle),photonDist*0.3*Math.sin(photonAngle*3),photonDist*Math.sin(photonAngle));
for(var i=0;i<photonTrail.length;i++){var pd=Math.max(0,photonDist-i*0.06);photonTrail[i].position.set(pd*Math.cos(photonAngle-i*0.05),pd*0.3*Math.sin((photonAngle-i*0.05)*3),pd*Math.sin(photonAngle-i*0.05));}
if(photonDist>15)photonActive=false;}
var pulse=1+0.05*Math.sin(Date.now()*0.003);nucleus.scale.set(pulse,pulse,pulse);
controls.update();renderer.render(scene,camera);}
animate();
window.addEventListener('resize',function(){w=container.clientWidth;h=container.clientHeight;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);});
})();
</script>"""
    html = html.replace("__RADII__", json.dumps(orbit_radii))
    html = html.replace("__COLORS__", json.dumps(orbit_colors))
    html = html.replace("__CN__", str(min(current_n,5)-1))
    html = html.replace("__ENERGIES__", json.dumps(energy_vals))
    html = html.replace("__TRANS_CODE__", trans_code)
    html = html.replace("__PHOTON_CODE__", photon_code)
    html = html.replace("__SOUND_CODE__", sound_code)
    return html

# ============================================================
# ENERGY LEVEL DIAGRAM
# ============================================================
def energy_level_diagram(highlight_ni=None, highlight_nf=None, transition_type="emit"):
    ni_str = str(highlight_ni or 0)
    nf_str = str(highlight_nf or 0)
    html = """<canvas id="eld" width="700" height="450" style="width:100%;max-width:700px;border-radius:12px;background:#0a0e1a;border:1px solid rgba(0,229,255,0.15);"></canvas>
<script>
(function(){
var c=document.getElementById('eld');var ctx=c.getContext('2d');var W=c.width,H=c.height;
var levels=[{n:1,e:-13.6,y:380,color:'#ff3366'},{n:2,e:-3.4,y:290,color:'#00ff88'},{n:3,e:-1.51,y:240,color:'#3388ff'},{n:4,e:-0.85,y:210,color:'#ffcc00'},{n:5,e:-0.544,y:190,color:'#cc66ff'},{n:6,e:-0.378,y:175,color:'#ff8844'}];
var ni=__NI__;var nf=__NF__;var tType='__TT__';
ctx.fillStyle='#0a0e1a';ctx.fillRect(0,0,W,H);
ctx.strokeStyle='rgba(0,229,255,0.05)';ctx.lineWidth=1;
for(var x=0;x<W;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
for(var y=0;y<H;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
levels.forEach(function(lv){
ctx.strokeStyle=lv.color;ctx.lineWidth=(lv.n===ni||lv.n===nf)?4:2;ctx.globalAlpha=(lv.n===ni||lv.n===nf)?1.0:0.6;
ctx.beginPath();ctx.moveTo(80,lv.y);ctx.lineTo(350,lv.y);ctx.stroke();ctx.globalAlpha=1.0;
ctx.font='bold 16px monospace';ctx.fillStyle=lv.color;ctx.textAlign='right';ctx.fillText('n = '+lv.n,72,lv.y+5);
ctx.textAlign='left';ctx.fillText(lv.e.toFixed(2)+' eV',360,lv.y+5);
});
ctx.strokeStyle='#ffffff';ctx.lineWidth=1;ctx.setLineDash([5,5]);ctx.beginPath();ctx.moveTo(80,150);ctx.lineTo(350,150);ctx.stroke();ctx.setLineDash([]);
ctx.font='14px monospace';ctx.fillStyle='#94a3b8';ctx.textAlign='right';ctx.fillText('n = inf',72,155);ctx.textAlign='left';ctx.fillText('0 eV',360,155);
if(ni>0&&nf>0&&ni!==nf){
var fromLv=levels.find(function(l){return l.n===ni;});var toLv=levels.find(function(l){return l.n===nf;});
if(fromLv&&toLv){var ax=220;ctx.strokeStyle=tType==='emit'?'#ff6b35':'#22d3ee';ctx.fillStyle=tType==='emit'?'#ff6b35':'#22d3ee';ctx.lineWidth=3;
ctx.beginPath();ctx.moveTo(ax,fromLv.y);ctx.lineTo(ax,toLv.y);ctx.stroke();
var dir=tType==='emit'?1:-1;ctx.beginPath();ctx.moveTo(ax,toLv.y);ctx.lineTo(ax-8,toLv.y-dir*12);ctx.lineTo(ax+8,toLv.y-dir*12);ctx.closePath();ctx.fill();
var midY=(fromLv.y+toLv.y)/2;ctx.strokeStyle=tType==='emit'?'#ffcc00':'#00ffcc';ctx.lineWidth=2;ctx.beginPath();
for(var i=0;i<60;i++){var px=ax+30+i*1.5;var py=midY+Math.sin(i*0.5)*8;if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py);}ctx.stroke();
ctx.font='bold 13px sans-serif';ctx.fillStyle=tType==='emit'?'#ffcc00':'#00ffcc';ctx.textAlign='left';ctx.fillText(tType==='emit'?'photon emitted':'photon absorbed',ax+35,midY-15);
var dE=Math.abs(fromLv.e-toLv.e).toFixed(2);ctx.font='bold 14px monospace';ctx.fillStyle='#ffffff';ctx.textAlign='center';ctx.fillText('dE = '+dE+' eV',ax+90,midY+25);
}}
ctx.font='bold 16px sans-serif';ctx.fillStyle='#00e5ff';ctx.textAlign='center';ctx.fillText('Hydrogen Energy Levels',W/2,30);
ctx.font='12px monospace';ctx.fillStyle='#a855f7';ctx.textAlign='right';ctx.fillText('Ionization = 13.6 eV',72,135);
})();
</script>"""
    html = html.replace("__NI__", ni_str)
    html = html.replace("__NF__", nf_str)
    html = html.replace("__TT__", transition_type)
    return html

# ============================================================
# SPECTRUM VISUALIZATION
# ============================================================
def spectrum_visualization(spec_type="emission", series="balmer"):
    html = """<canvas id="specCanvas" width="800" height="300" style="width:100%;max-width:800px;border-radius:12px;background:#000;border:1px solid rgba(0,229,255,0.15);"></canvas>
<script>
(function(){
var c=document.getElementById('specCanvas');var ctx=c.getContext('2d');var W=c.width,H=c.height;
var specType='__ST__';var series='__SR__';
var lines={lyman:[121.6,102.6,97.3,95.0,93.8],balmer:[656.3,486.1,434.0,410.2,397.0],paschen:[1875,1282,1094,1005,954],brackett:[4051,2625,2166,1945]};
function wlToRGB(wl){var r=0,g=0,b=0;if(wl>=380&&wl<440){r=-(wl-440)/(440-380);b=1.0;}else if(wl>=440&&wl<490){g=(wl-440)/(490-440);b=1.0;}else if(wl>=490&&wl<510){g=1.0;b=-(wl-510)/(510-490);}else if(wl>=510&&wl<580){r=(wl-510)/(580-510);g=1.0;}else if(wl>=580&&wl<645){r=1.0;g=-(wl-645)/(645-580);}else if(wl>=645&&wl<=780){r=1.0;}var f=1.0;if(wl>=380&&wl<420)f=0.3+0.7*(wl-380)/(420-380);else if(wl>=645&&wl<=780)f=0.3+0.7*(780-wl)/(780-645);return[Math.round(r*f*255),Math.round(g*f*255),Math.round(b*f*255)];}
ctx.fillStyle='#000';ctx.fillRect(0,0,W,H);var margin=40;var plotW=W-2*margin;var plotH=H-80;var minWL=90,maxWL=2000;
if(series==='balmer'){minWL=380;maxWL=700;}else if(series==='lyman'){minWL=90;maxWL=130;}else if(series==='paschen'){minWL=900;maxWL=2000;}
function wlToX(wl){return margin+(wl-minWL)/(maxWL-minWL)*plotW;}
if(specType==='emission'){var lns=lines[series]||lines.balmer;lns.forEach(function(wl){var x=wlToX(wl);var rgb=wlToRGB(wl);var col=wl>=380&&wl<=780?'rgb('+rgb[0]+','+rgb[1]+','+rgb[2]+')':'#ff4444';ctx.shadowColor=col;ctx.shadowBlur=15;ctx.fillStyle=col;ctx.fillRect(x-2,30,4,plotH);ctx.shadowBlur=0;ctx.font='11px monospace';ctx.fillStyle='#ccc';ctx.textAlign='center';ctx.fillText(wl.toFixed(1)+'nm',x,H-30);});
}else if(specType==='absorption'){for(var px=margin;px<margin+plotW;px++){var wl=minWL+(px-margin)/plotW*(maxWL-minWL);var rgb=wlToRGB(wl);var col=wl>=380&&wl<=780?'rgb('+rgb[0]+','+rgb[1]+','+rgb[2]+')':'rgba(50,0,0,0.5)';ctx.fillStyle=col;ctx.fillRect(px,30,1,plotH);}var lns=lines[series]||lines.balmer;lns.forEach(function(wl){var x=wlToX(wl);ctx.fillStyle='#000';ctx.fillRect(x-2,30,4,plotH);ctx.font='11px monospace';ctx.fillStyle='#ccc';ctx.textAlign='center';ctx.fillText(wl.toFixed(1)+'nm',x,H-30);});
}else{for(var px=margin;px<margin+plotW;px++){var wl=minWL+(px-margin)/plotW*(maxWL-minWL);var rgb=wlToRGB(wl);var col=wl>=380&&wl<=780?'rgb('+rgb[0]+','+rgb[1]+','+rgb[2]+')':'rgba(50,0,0,0.5)';ctx.fillStyle=col;ctx.fillRect(px,30,1,plotH);}}
ctx.font='12px monospace';ctx.fillStyle='#94a3b8';ctx.textAlign='center';ctx.fillText('Wavelength (nm)',W/2,H-8);
var sn={lyman:'Lyman (UV)',balmer:'Balmer (Visible)',paschen:'Paschen (IR)',brackett:'Brackett (IR)'};
var tn={emission:'Emission Line Spectrum',absorption:'Absorption Line Spectrum',continuous:'Continuous Spectrum'};
ctx.font='bold 14px sans-serif';ctx.fillStyle='#00e5ff';ctx.textAlign='center';ctx.fillText(tn[specType]+' | '+sn[series],W/2,20);
})();
</script>"""
    html = html.replace("__ST__", spec_type)
    html = html.replace("__SR__", series)
    return html

# ============================================================
# DIFFRACTION PATTERN
# ============================================================
def diffraction_pattern(wavelength_nm=500, slit_width_nm=1000, particle_type="light"):
    html = """<canvas id="diffCanvas" width="800" height="400" style="width:100%;max-width:800px;border-radius:12px;background:#000;border:1px solid rgba(0,229,255,0.15);"></canvas>
<script>
(function(){
var c=document.getElementById('diffCanvas');var ctx=c.getContext('2d');var W=c.width,H=c.height;
var wl=__WL__;var sw=__SW__;var pType='__PT__';
function wlToRGB2(w){var r=0,g=0,b=0;if(w>=380&&w<440){r=-(w-440)/(440-380);b=1;}else if(w>=440&&w<490){g=(w-440)/(490-440);b=1;}else if(w>=490&&w<510){g=1;b=-(w-510)/(510-490);}else if(w>=510&&w<580){r=(w-510)/(580-510);g=1;}else if(w>=580&&w<645){r=1;g=-(w-645)/(645-580);}else if(w>=645&&w<=780){r=1;}var f=1;if(w>=380&&w<420)f=0.3+0.7*(w-380)/(420-380);else if(w>=645&&w<=780)f=0.3+0.7*(780-w)/(780-645);return[Math.round(r*f*255),Math.round(g*f*255),Math.round(b*f*255)];}
ctx.fillStyle='#000';ctx.fillRect(0,0,W,H);
var useWl=pType==='electron'?420:wl;var rgb=wlToRGB2(useWl);var colStr='rgb('+rgb[0]+','+rgb[1]+','+rgb[2]+')';
var patternH=300;var startY=50;var ratio=wl/sw;
for(var py=0;py<patternH;py++){var y=(py-patternH/2)/(patternH/2);var sinT=y*0.3;var beta=Math.PI*sinT/ratio;var intensity;if(Math.abs(beta)<0.0001)intensity=1;else intensity=Math.pow(Math.sin(beta)/beta,2);var barW=intensity*500;ctx.fillStyle=colStr;ctx.globalAlpha=intensity;ctx.fillRect(100,startY+py,barW,1);ctx.globalAlpha=1;}
for(var py=0;py<patternH;py++){var y=(py-patternH/2)/(patternH/2);var sinT=y*0.3;var beta=Math.PI*sinT/ratio;var intensity;if(Math.abs(beta)<0.0001)intensity=1;else intensity=Math.pow(Math.sin(beta)/beta,2);ctx.fillStyle=colStr;ctx.globalAlpha=intensity;ctx.fillRect(680,startY+py,40,1);ctx.globalAlpha=1;}
ctx.strokeStyle='#333';ctx.strokeRect(680,startY,40,patternH);
ctx.font='bold 14px sans-serif';ctx.fillStyle='#00e5ff';ctx.textAlign='center';
var title=pType==='electron'?'Electron Diffraction Pattern':'Light Diffraction Pattern';ctx.fillText(title,W/2,25);
ctx.font='12px monospace';ctx.fillStyle='#a855f7';var lamStr=pType==='electron'?'0.027 nm':wl+' nm';ctx.fillText('lambda = '+lamStr+'  |  a = '+sw+' nm',W/2,H-15);
})();
</script>"""
    html = html.replace("__WL__", str(wavelength_nm))
    html = html.replace("__SW__", str(slit_width_nm))
    html = html.replace("__PT__", particle_type)
    return html

# ============================================================
# X-RAY SPECTRUM
# ============================================================
def xray_spectrum(min_wl_nm=0.02, z=74):
    html = """<canvas id="xrayCanvas" width="750" height="350" style="width:100%;max-width:750px;border-radius:12px;background:#0a0e1a;border:1px solid rgba(0,229,255,0.15);"></canvas>
<script>
(function(){
var c=document.getElementById('xrayCanvas');var ctx=c.getContext('2d');var W=c.width,H=c.height;var minWL=__MWL__;
ctx.fillStyle='#0a0e1a';ctx.fillRect(0,0,W,H);
ctx.strokeStyle='rgba(0,229,255,0.05)';ctx.lineWidth=1;
for(var x=60;x<W-20;x+=50){ctx.beginPath();ctx.moveTo(x,40);ctx.lineTo(x,H-60);ctx.stroke();}
for(var y=40;y<H-60;y+=40){ctx.beginPath();ctx.moveTo(60,y);ctx.lineTo(W-20,y);ctx.stroke();}
var plotL=60,plotR=W-20,plotT=40,plotB=H-60,plotW=plotR-plotL,plotH=plotB-plotT;var wlMin=0.01,wlMax=0.15;
function wlToX(wl){return plotL+(wl-wlMin)/(wlMax-wlMin)*plotW;}
ctx.beginPath();ctx.moveTo(wlToX(minWL),plotB);var maxI=0;var points=[];
for(var i=0;i<=200;i++){var wl=minWL+(wlMax-minWL)*i/200;var intensity=(1/minWL-1/wl);if(intensity<0)intensity=0;points.push({wl:wl,i:intensity});if(intensity>maxI)maxI=intensity;}
points.forEach(function(p){var x=wlToX(p.wl);var y=plotB-(p.i/maxI)*plotH*0.7;ctx.lineTo(x,y);});
ctx.lineTo(wlToX(wlMax),plotB);ctx.closePath();
var grad=ctx.createLinearGradient(0,plotT,0,plotB);grad.addColorStop(0,'rgba(0,229,255,0.3)');grad.addColorStop(1,'rgba(0,229,255,0.02)');
ctx.fillStyle=grad;ctx.fill();ctx.strokeStyle='#00e5ff';ctx.lineWidth=2;ctx.stroke();
[0.021,0.018].forEach(function(wl,idx){var x=wlToX(wl);ctx.strokeStyle=idx===0?'#ff6b35':'#ffcc00';ctx.lineWidth=3;ctx.shadowColor=ctx.strokeStyle;ctx.shadowBlur=10;ctx.beginPath();ctx.moveTo(x,plotB);ctx.lineTo(x,plotT+20);ctx.stroke();ctx.shadowBlur=0;ctx.font='bold 12px monospace';ctx.fillStyle=ctx.strokeStyle;ctx.textAlign='center';ctx.fillText(idx===0?'Ka':'Kb',x,plotT+15);});
var minX=wlToX(minWL);ctx.strokeStyle='#a855f7';ctx.lineWidth=2;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(minX,plotT);ctx.lineTo(minX,plotB);ctx.stroke();ctx.setLineDash([]);
ctx.font='11px monospace';ctx.fillStyle='#a855f7';ctx.textAlign='center';ctx.fillText('lambda_min',minX,plotB+15);
ctx.font='bold 14px sans-serif';ctx.fillStyle='#00e5ff';ctx.textAlign='center';ctx.fillText('X-Ray Spectrum',W/2,22);
})();
</script>"""
    html = html.replace("__MWL__", str(min_wl_nm))
    return html

# ============================================================
# X-RAY TUBE DIAGRAM
# ============================================================
def xray_tube_diagram():
    return """<canvas id="xrayTube" width="700" height="350" style="width:100%;max-width:700px;border-radius:12px;background:#0a0e1a;border:1px solid rgba(0,229,255,0.15);"></canvas>
<script>
(function(){
var c=document.getElementById('xrayTube');var ctx=c.getContext('2d');var W=c.width,H=c.height;
ctx.fillStyle='#0a0e1a';ctx.fillRect(0,0,W,H);
ctx.strokeStyle='#334466';ctx.lineWidth=3;ctx.beginPath();ctx.ellipse(350,175,280,130,0,0,Math.PI*2);ctx.stroke();ctx.fillStyle='rgba(10,14,26,0.8)';ctx.fill();
ctx.strokeStyle='#ff6b35';ctx.lineWidth=3;ctx.beginPath();for(var i=0;i<5;i++){var y=140+i*15;ctx.moveTo(100,y);ctx.quadraticCurveTo(120,y+8,100,y+15);}ctx.stroke();
var glowGrad=ctx.createRadialGradient(110,175,5,110,175,40);glowGrad.addColorStop(0,'rgba(255,107,53,0.4)');glowGrad.addColorStop(1,'rgba(255,107,53,0)');ctx.fillStyle=glowGrad;ctx.fillRect(70,135,80,80);
ctx.fillStyle='#445577';ctx.fillRect(560,130,30,90);ctx.fillStyle='#556688';ctx.fillRect(555,125,40,10);ctx.fillRect(555,215,40,10);
ctx.fillStyle='#00e5ff';for(var i=0;i<15;i++){var x=150+i*25;var y=175+Math.sin(i*0.8)*5;ctx.globalAlpha=1-i*0.05;ctx.beginPath();ctx.arc(x,y,Math.max(1,3-i*0.1),0,Math.PI*2);ctx.fill();}ctx.globalAlpha=1;
ctx.strokeStyle='#ffcc00';ctx.lineWidth=1.5;for(var i=0;i<8;i++){var angle=-0.6+i*0.17;ctx.beginPath();ctx.moveTo(590,175);var len=60+Math.random()*20;ctx.lineTo(590+Math.cos(angle)*len,175+Math.sin(angle)*len);ctx.stroke();}
ctx.fillStyle='rgba(255,204,0,0.15)';ctx.fillRect(615,155,10,40);ctx.strokeStyle='#ffcc00';ctx.lineWidth=2;ctx.strokeRect(615,155,10,40);
ctx.font='bold 13px sans-serif';ctx.textAlign='center';ctx.fillStyle='#ff6b35';ctx.fillText('Cathode',110,120);ctx.fillStyle='#4488ff';ctx.fillText('Anode',575,120);ctx.fillStyle='#00e5ff';ctx.fillText('e-',350,160);ctx.fillStyle='#ffcc00';ctx.fillText('X-rays',640,148);
ctx.fillStyle='#222';ctx.fillRect(630,100,15,50);ctx.fillRect(630,200,15,50);ctx.font='10px sans-serif';ctx.fillStyle='#666';ctx.fillText('Pb',637,265);
ctx.font='bold 14px sans-serif';ctx.fillStyle='#00e5ff';ctx.fillText('X-Ray Tube',350,25);
ctx.strokeStyle='#ff3366';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(110,260);ctx.lineTo(110,290);ctx.stroke();ctx.beginPath();ctx.moveTo(575,260);ctx.lineTo(575,290);ctx.stroke();ctx.beginPath();ctx.moveTo(110,290);ctx.lineTo(575,290);ctx.stroke();
ctx.font='bold 12px monospace';ctx.fillStyle='#ff3366';ctx.textAlign='center';ctx.fillText('High Voltage (dV)',342,285);
ctx.font='bold 20px monospace';ctx.fillText('+',575,275);ctx.fillText('-',110,275);
})();
</script>"""

# ============================================================
# LASER DIAGRAM
# ============================================================
def laser_diagram():
    return """<canvas id="laserCanvas" width="750" height="300" style="width:100%;max-width:750px;border-radius:12px;background:#0a0e1a;border:1px solid rgba(0,229,255,0.15);"></canvas>
<script>
(function(){
var c=document.getElementById('laserCanvas');var ctx=c.getContext('2d');var W=c.width,H=c.height;
ctx.fillStyle='#0a0e1a';ctx.fillRect(0,0,W,H);
ctx.strokeStyle='#334466';ctx.lineWidth=4;ctx.strokeRect(150,80,450,140);
var medGrad=ctx.createLinearGradient(200,0,550,0);medGrad.addColorStop(0,'rgba(255,50,50,0.15)');medGrad.addColorStop(0.5,'rgba(255,50,50,0.3)');medGrad.addColorStop(1,'rgba(255,50,50,0.15)');ctx.fillStyle=medGrad;ctx.fillRect(200,85,350,130);
ctx.fillStyle='#4488aa';ctx.fillRect(150,80,8,140);ctx.fillRect(592,80,8,140);
ctx.fillStyle='#ff3333';[[250,150],[450,160],[350,140]].forEach(function(p){ctx.beginPath();ctx.arc(p[0],p[1],4,0,Math.PI*2);ctx.fill();});
for(var i=0;i<5;i++){ctx.strokeStyle='rgba(255,0,0,'+(0.8-i*0.15)+')';ctx.lineWidth=3-i*0.5;ctx.beginPath();ctx.moveTo(600,150);ctx.lineTo(720,150);ctx.stroke();}
var outGlow=ctx.createRadialGradient(600,150,2,600,150,30);outGlow.addColorStop(0,'rgba(255,0,0,0.6)');outGlow.addColorStop(1,'rgba(255,0,0,0)');ctx.fillStyle=outGlow;ctx.fillRect(570,120,60,60);
ctx.font='bold 13px sans-serif';ctx.textAlign='center';ctx.fillStyle='#66aacc';ctx.fillText('Full Mirror',154,70);ctx.fillText('Partial Mirror',596,70);
ctx.fillStyle='#ff5555';ctx.fillText('Active Medium',375,230);ctx.fillStyle='#ff0000';ctx.font='bold 14px sans-serif';ctx.fillText('LASER Beam ->',660,140);
ctx.strokeStyle='#22d3ee';ctx.lineWidth=2;ctx.setLineDash([4,3]);for(var i=0;i<5;i++){var x=240+i*70;ctx.beginPath();ctx.moveTo(x,40);ctx.lineTo(x,80);ctx.stroke();ctx.beginPath();ctx.moveTo(x-4,75);ctx.lineTo(x,85);ctx.lineTo(x+4,75);ctx.stroke();}ctx.setLineDash([]);
ctx.font='11px sans-serif';ctx.fillStyle='#22d3ee';ctx.fillText('Pump Energy',375,30);
ctx.font='bold 14px sans-serif';ctx.fillStyle='#00e5ff';ctx.fillText('How LASER Works',375,H-15);
})();
</script>"""

# ============================================================
# DE BROGLIE WAVE
# ============================================================
def de_broglie_wave(mass_kg=9.11e-31, velocity=1e6):
    html = """<canvas id="deBroglieCanvas" width="750" height="300" style="width:100%;max-width:750px;border-radius:12px;background:#0a0e1a;border:1px solid rgba(0,229,255,0.15);"></canvas>
<script>
(function(){
var c=document.getElementById('deBroglieCanvas');var ctx=c.getContext('2d');var W=c.width,H=c.height;
var mass=__MASS__;var vel=__VEL__;var h=6.63e-34;var lambda=h/(mass*vel);
ctx.fillStyle='#0a0e1a';ctx.fillRect(0,0,W,H);
var particleX=100;var particleY=H/2;
ctx.fillStyle='#00e5ff';ctx.shadowColor='#00e5ff';ctx.shadowBlur=20;ctx.beginPath();ctx.arc(particleX,particleY,12,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;
var displayLambda=Math.min(80,Math.max(15,lambda*1e10*0.1));
ctx.strokeStyle='#a855f7';ctx.lineWidth=2.5;ctx.beginPath();
for(var x=particleX+15;x<W-30;x++){var phase=(x-particleX)/displayLambda*Math.PI*2;var y=particleY+Math.sin(phase)*60;if(x===particleX+15)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.stroke();
ctx.globalAlpha=0.1;ctx.fillStyle='#a855f7';ctx.beginPath();ctx.moveTo(particleX+15,particleY);
for(var x=particleX+15;x<W-30;x++){var phase=(x-particleX)/displayLambda*Math.PI*2;var y=particleY+Math.sin(phase)*60;ctx.lineTo(x,y);}
ctx.lineTo(W-30,particleY);ctx.closePath();ctx.fill();ctx.globalAlpha=1;
var markX=particleX+20;ctx.strokeStyle='#ff6b35';ctx.lineWidth=2;
ctx.beginPath();ctx.moveTo(markX,particleY+75);ctx.lineTo(markX,particleY+85);ctx.stroke();
ctx.beginPath();ctx.moveTo(markX+displayLambda,particleY+75);ctx.lineTo(markX+displayLambda,particleY+85);ctx.stroke();
ctx.beginPath();ctx.moveTo(markX,particleY+80);ctx.lineTo(markX+displayLambda,particleY+80);ctx.stroke();
ctx.font='bold 13px monospace';ctx.fillStyle='#ff6b35';ctx.textAlign='center';ctx.fillText('lambda',markX+displayLambda/2,particleY+98);
ctx.strokeStyle='#22d3ee';ctx.lineWidth=2;
ctx.beginPath();ctx.moveTo(particleX+15,particleY-25);ctx.lineTo(particleX+70,particleY-25);ctx.stroke();
ctx.beginPath();ctx.moveTo(particleX+70,particleY-25);ctx.lineTo(particleX+62,particleY-30);ctx.stroke();
ctx.beginPath();ctx.moveTo(particleX+70,particleY-25);ctx.lineTo(particleX+62,particleY-20);ctx.stroke();
ctx.font='12px monospace';ctx.fillStyle='#22d3ee';ctx.fillText('v',particleX+42,particleY-30);
ctx.font='bold 14px sans-serif';ctx.fillStyle='#00e5ff';ctx.textAlign='center';ctx.fillText('De Broglie Matter Wave',W/2,25);
var lambdaStr;if(lambda<1e-9)lambdaStr=(lambda*1e12).toFixed(2)+' pm';else if(lambda<1e-6)lambdaStr=(lambda*1e9).toFixed(3)+' nm';else lambdaStr=lambda.toExponential(2)+' m';
ctx.font='bold 13px monospace';ctx.fillStyle='#ff6b35';ctx.textAlign='center';ctx.fillText('lambda = '+lambdaStr,W/2,H-10);
})();
</script>"""
    html = html.replace("__MASS__", str(mass_kg))
    html = html.replace("__VEL__", str(velocity))
    return html

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0;">
        <div style="font-size:2.5em;">⚛️</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📋 الأقسام")
    section = st.radio(
        "",
        [
            "🏠 الرئيسية",
            "⚛️ نموذج بور",
            "📊 مستويات الطاقة والمعادلات",
            "💡 طاقة الفوتون",
            "🌈 الأطياف الذرية",
            "📐 الطيف الهيدروجيني",
            "🌊 الطبيعة المزدوجة",
            "🔬 فرضية دي بروي",
            "🏥 الأشعة السينية",
            "⚡ تطبيقات تكنولوجية",
        ],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="author-badge">', unsafe_allow_html=True)
    st.markdown("<p>إعداد: Israa Youssuf Samara</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# HOME PAGE
# ============================================================
if section == "🏠 الرئيسية":
    st.markdown('<h1 class="section-title">التركيب الذري | Atomic Structure</h1>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card card-accent" style="text-align:center;padding:40px;">
        <div style="font-size:4em;margin-bottom:16px;">⚛️</div>
        <h2 style="color:#00e5ff;margin-bottom:12px;">رحلة داخل الذرة</h2>
        <p style="color:#94a3b8;font-size:1.1em;max-width:700px;margin:0 auto;">
            تطبيق تعليمي تفاعلي يستخدم الرسوم المتحركة ثلاثية الأبعاد والأدوات التفاعلية
            لشرح نموذج بور لذرة الهيدروجين والأطياف الذرية والطبيعة المزدوجة وفرضية دي بروي
            والأشعة السينية وتطبيقات الليزر.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:2em;">🔬</div>
            <h4>رسوم ثلاثية الأبعاد</h4>
            <p style="color:#94a3b8;font-size:0.85em;">نماذج 3D تفاعلية لذرة الهيدروجين ونمط الحيود</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:2em;">🧮</div>
            <h4>حاسبات تفاعلية</h4>
            <p style="color:#94a3b8;font-size:0.85em;">حساب الطاقة والطول الموجي والتردد والزخم</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:2em;">🔊</div>
            <h4>تأثيرات صوتية</h4>
            <p style="color:#94a3b8;font-size:0.85em;">أصوات تمثل انتقالات الإلكترون بين المستويات</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    topics = [
        ("⚛️", "نموذج بور", "Bohr's Model", "فرضيات بور الأربع لذرة الهيدروجين مع رسوم 3D متحركة"),
        ("📊", "مستويات الطاقة", "Energy Levels", "المعادلات الرياضية وطاقة التأين والزخم الزاوي"),
        ("💡", "طاقة الفوتون", "Photon Energy", "حساب طاقة الفوتون المنبعث والممتص تفاعلياً"),
        ("🌈", "الأطياف الذرية", "Atomic Spectra", "طيف الانبعاث والامتصاص الخطي والمتصل"),
        ("📐", "الطيف الهيدروجيني", "H Spectrum", "اشتقاق معادلة ريدبرغ والتحقق التجريبي"),
        ("🌊", "الطبيعة المزدوجة", "Duality", "الطبيعة الموجية-الجسيمية للإشعاع والمادة"),
        ("🔬", "فرضية دي بروي", "De Broglie", "الاشتقاق الرياضي والحسابات التفاعلية"),
        ("🏥", "الأشعة السينية", "X-Rays", "أنبوب الأشعة السينية والطيف والاستخدامات الطبية"),
        ("⚡", "تطبيقات تكنولوجية", "Applications", "الليزر والأشعة السينية في الحياة اليومية"),
    ]
    for icon, ar_title, en_title, desc in topics:
        st.markdown(f"""
        <div class="card" style="display:flex;align-items:center;gap:16px;padding:16px 20px;">
            <div style="font-size:1.8em;min-width:50px;text-align:center;">{icon}</div>
            <div>
                <h4 style="margin:0 0 4px 0;">{ar_title} <span style="color:#94a3b8;font-weight:400;font-size:0.85em;">| {en_title}</span></h4>
                <p style="margin:0;color:#94a3b8;font-size:0.85em;">{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# SECTION 1: BOHR'S MODEL
# ============================================================
elif section == "⚛️ نموذج بور":
    st.markdown('<h1 class="section-title">نموذج بور لذرة الهيدروجين | Bohr\'s Model</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
        <h4 style="color:#ff6b35;margin:0 0 8px 0;">💡 من الحياة اليومية</h4>
        <p style="margin:0;">
        تخيّل أنك في مبنى سكني المصعد فيه لا يقف عند كل طابق بل عند طوابق محددة فقط (مثلاً الطابق 1 و3 و5).
        لا يمكنك الوقوف بين طابقين! الإلكترون في ذرة الهيدروجين يعمل بنفس الطريقة — لا يستطيع البقاء
        في أي مكان بين مستويات الطاقة، بل في مستويات محددة فقط. وكما تحتاج طاقة معينة للصعود من طابق لآخر،
        فإن الإلكترون يحتاج طاقة محددة (فوتون) للانتقال بين المستويات.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">🔬 الخلفية التاريخية</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <p>بعد تجربة <strong>رذرفورد (1911)</strong> التي أثبتت وجود نواة موجبة الشحنة صغيرة جداً،
        ظهرت مشكلة كبيرة: الإلكترون المشحون يدور حول النواة ويتسارع، وبحسب الفيزياء الكلاسيكية يجب أن يشع
        طاقة باستمرار ويسقط على النواة! لكن الذرات مستقرة في الواقع.</p>
        <p>العالم <strong>نيلز بور (1913)</strong> حلّ هذه المشكلة بفرضيات ثورية أعادت فهمنا للذرة.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">📋 فرضيات بور الأربع</h3>', unsafe_allow_html=True)
    postulates = [
        ("الفرضية الأولى: المدارات الدائرية", "يدور الإلكترون حول البروتون في مسارات دائرية تحت تأثير قوة التجاذب الكهربائي.", "L = m_e * v * r"),
        ("الفرضية الثانية: التكمية (الاستقرار)", "للإلكترون مدارات مسموحة محددة (مستويات طاقة) يحتلها، وإذا بقي في المستوى نفسه فلا يشع طاقة ولا يمتصها.", "E_n = -13.6 / n^2  (eV)"),
        ("الفرضية الثالثة: انتقال الطاقة", "يشع أو يمتص الإلكترون طاقة فقط عند الانتقال بين مستويين. طاقة الفوتون = الفرق بين الطاقتين.", "E = |E_f - E_i| = hf"),
        ("الفرضية الرابعة: تكمية الزخم الزاوي", "الزخم الزاوي للإلكترون يساوي مضاعفات صحيحة من h-bar.", "L = n * h-bar = n*(h/2pi)"),
    ]
    clr = ['#ff3366','#00ff88','#3388ff','#ffcc00']
    for i, (title, desc, eq) in enumerate(postulates):
        st.markdown(f"""
        <div class="card" style="border-left:4px solid {clr[i]};">
            <h4 style="margin:0 0 8px 0;color:{clr[i]};">{i+1}. {title}</h4>
            <p style="margin:0 0 10px 0;">{desc}</p>
            <div class="equation-box">{eq}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-title">🌐 النموذج ثلاثي الأبعاد التفاعلي</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        n_level = st.selectbox("اختر مستوى الطاقة الحالي (n):", [1,2,3,4,5], index=0, key="bohr_n_select")
        st.markdown("**انتقال الإلكترون إلى:**")
        tc1, tc2 = st.columns(2)
        with tc1:
            do_transition = st.button("⚡ شغّل الانتقال", type="primary")
        with tc2:
            trans_target = st.selectbox("المستوى المستهدف:", [1,2,3,4,5], key="trans_target")

        if do_transition:
            is_emission = trans_target < n_level
            components.html(bohr_atom_3d(current_n=n_level, transition_to=trans_target, emit=is_emission), height=520)
        else:
            components.html(bohr_atom_3d(current_n=n_level), height=520)

    with col2:
        st.markdown("""
        <div class="info-box">
            <h4 style="color:#a855f7;margin:0 0 8px 0;">🎮 كيفية التفاعل</h4>
            <ul style="margin:0;padding-right:20px;color:#94a3b8;font-size:0.9em;">
                <li>اسحب بالماوس للتدوير</li>
                <li>مرّر العجلة للتقريب/التبعيد</li>
                <li>اختر المستوى والانتقال</li>
                <li>استمع لصوت الانتقال!</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card" style="margin-top:16px;">
            <h4 style="color:#00e5ff;margin:0 0 8px 0;">📊 بيانات المدار الحالي</h4>
            <p style="margin:4px 0;"><strong style="color:#ff3366;">n = {n_level}</strong></p>
            <p style="margin:4px 0;"><strong>E = {calc_energy_level(n_level):.3f} eV</strong></p>
            <p style="margin:4px 0;"><strong>L = {n_level}h-bar = {calc_angular_momentum(n_level):.3e} J.s</strong></p>
        </div>
        """, unsafe_allow_html=True)

        if do_transition and trans_target != n_level:
            de = calc_photon_energy(n_level, trans_target)
            wl = calc_wavelength(de)
            freq = calc_frequency(de)
            region = "UV" if wl < 380e-9 else ("مرئي" if wl < 780e-9 else "IR")
            st.markdown(f"""
            <div class="card card-accent" style="margin-top:16px;">
                <h4 style="color:#ff6b35;margin:0 0 8px 0;">⚡ بيانات الفوتون</h4>
                <p style="margin:4px 0;">dE = <strong>{de:.4f} eV</strong></p>
                <p style="margin:4px 0;">lambda = <strong>{wl*1e9:.2f} nm</strong> ({region})</p>
                <p style="margin:4px 0;">f = <strong>{freq:.3e} Hz</strong></p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# SECTION 2: ENERGY LEVELS & EQUATIONS
# ============================================================
elif section == "📊 مستويات الطاقة والمعادلات":
    st.markdown('<h1 class="section-title">مستويات الطاقة والمعادلات الرياضية</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
        <h4 style="color:#ff6b35;margin:0 0 8px 0;">💡 من الحياة اليومية</h4>
        <p style="margin:0;">
        تخيّل سلّماً لا تستطيع الوقوف على أي درجة منه فحسب، بل كل درجة لها ارتفاع محدد بدقة.
        المستوى الأول (الأرضية) طاقته -13.6 eV، والمستوى الثاني طاقته -3.4 eV وهكذا.
        كلما صعدت تصبح الفوارق بين الدرجات أصغر، حتى تصل للسطح (n = inf) حيث الطاقة = 0.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">📐 معادلة طاقة المستوى</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <p style="margin-bottom:12px;">تُعبَّر عن طاقة الإلكترون في المستوى n لذرة الهيدروجين بوحدة إلكترون فولت:</p>
        <div class="equation-box" style="font-size:1.5em;">E_n = -13.6 / n^2  (eV)</div>
        <p style="margin-top:12px;">
        <strong style="color:#ff3366;">الإشارة السالبة</strong> تعني أن الإلكترون مقيد في الذرة.<br>
        <strong style="color:#00ff88;">n = 1</strong> هو مستوى الاستقرار (أقل طاقة = -13.6 eV).<br>
        <strong style="color:#3388ff;">n > 1</strong> هي مستويات الإثارة.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">🔄 الزخم الزاوي</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <p>فرض بور أن الزخم الزاوي للإلكترون مكمّم:</p>
        <div class="equation-box">L = n * h-bar = n * (h / 2pi)   حيث   h-bar = 1.055 x 10^-34 J.s</div>
        <ul style="padding-right:20px;">
            <li>n = 1:  L = h-bar = 1.055 x 10^-34 J.s</li>
            <li>n = 2:  L = 2*h-bar = 2.11 x 10^-34 J.s</li>
            <li>n = 3:  L = 3*h-bar = 3.165 x 10^-34 J.s</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">💡 معادلة طاقة الفوتون</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <p>عند انتقال الإلكترون بين مستويين:</p>
        <div class="equation-box">E = |E_f - E_i| = hf</div>
        <p><strong>E_i</strong>: طاقة المستوى الابتدائي | <strong>E_f</strong>: طاقة المستوى النهائي | <strong>f</strong>: تردد الفوتون | <strong>h</strong>: ثابت بلانك = 6.63 x 10^-34 J.s</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">⚡ طاقة التأين (Ionization Energy)</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card card-accent">
        <p style="margin-bottom:12px;"><strong style="color:#a855f7;font-size:1.1em;">طاقة التأين</strong> هي أقل طاقة لازمة لتحرير الإلكترون من الذرة تماماً (نقله من n=1 إلى n=inf) دون إكسابه طاقة حركية.</p>
        <div class="equation-box" style="font-size:1.3em;">E_ion = |E_inf - E_1| = |0 - (-13.6)| = <strong style="color:#ff6b35;">13.6 eV</strong></div>
        <div class="info-box" style="margin-top:12px;">
            <p style="margin:0;"><strong>ملاحظة:</strong> إذا زادت طاقة الفوتون الممتص عن 13.6 eV، يتحرر الإلكترون وتكون الزيادة طاقة حركية له.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-title">📊 مخطط مستويات الطاقة التفاعلي</h3>', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        sel_ni = st.selectbox("المستوى الابتدائي (n_i):", [1,2,3,4,5,6], index=2, key="eld_ni")
    with sc2:
        sel_nf = st.selectbox("المستوى النهائي (n_f):", [1,2,3,4,5,6], index=0, key="eld_nf")
    t_type = st.radio("نوع الانتقال:", ["انبعاث (إشعاع)", "امتصاص"], horizontal=True, key="eld_type")
    t_type_code = "emit" if "انبعاث" in t_type else "absorption"
    components.html(energy_level_diagram(highlight_ni=sel_ni, highlight_nf=sel_nf, transition_type=t_type_code), height=470)

    st.markdown('<h3 class="sub-title">📋 جدول مستويات الطاقة</h3>', unsafe_allow_html=True)
    rows = ""
    for n in range(1, 7):
        e = calc_energy_level(n)
        l = calc_angular_momentum(n)
        rows += f"<tr><td>{n}</td><td>{e:.3f}</td><td>{e*EV_TO_J:.3e}</td><td>{l:.3e}</td><td>{n}h-bar</td></tr>"
    st.markdown(f"""
    <style>
    .dt{{width:100%;border-collapse:collapse;font-size:0.9em;}}
    .dt th{{background:rgba(0,229,255,0.1);color:#00e5ff;padding:10px;border:1px solid rgba(0,229,255,0.15);text-align:center;font-family:monospace;}}
    .dt td{{padding:8px;text-align:center;border:1px solid rgba(0,229,255,0.08);color:#e0e0e0;font-family:monospace;}}
    </style>
    <table class="dt"><tr><th>n</th><th>E_n (eV)</th><th>E_n (J)</th><th>L (J.s)</th><th>L</th></tr>{rows}</table>
    """, unsafe_allow_html=True)

# ============================================================
# SECTION 3: PHOTON ENERGY CALCULATOR
# ============================================================
elif section == "💡 طاقة الفوتون":
    st.markdown('<h1 class="section-title">حاسبة طاقة الفوتون | Photon Energy Calculator</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
        <h4 style="color:#ff6b35;margin:0 0 8px 0;">💡 من الحياة اليومية</h4>
        <p style="margin:0;">عندما تضع كوب ماء في الميكروويف، تمتص جزيئات الماء فوتونات ذات تردد محدد (2.45 GHz)
        فتتذبذب وتسخّن الماء. كل فوتون له طاقة محددة = h x f. نفس المبدأ يحدث في الذرة.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<h3 class="sub-title">🔧 أدخل البيانات</h3>', unsafe_allow_html=True)
        calc_ni = st.selectbox("المستوى الابتدائي (n_i):", [1,2,3,4,5,6], index=2, key="calc_ni")
        calc_nf = st.selectbox("المستوى النهائي (n_f):", [1,2,3,4,5,6], index=0, key="calc_nf")

    with c2:
        e_i = calc_energy_level(calc_ni)
        e_f = calc_energy_level(calc_nf)
        dE = abs(e_f - e_i)
        wl = calc_wavelength(dE)
        freq = calc_frequency(dE)
        is_em = calc_nf < calc_ni

        region = "UV"
        color_box = "#8800ff"
        if wl >= 380e-9 and wl < 440e-9: region = "بنفسجي"; color_box = "#8800ff"
        elif wl >= 440e-9 and wl < 490e-9: region = "أزرق"; color_box = "#0066ff"
        elif wl >= 490e-9 and wl < 510e-9: region = "سماوي"; color_box = "#00ccff"
        elif wl >= 510e-9 and wl < 580e-9: region = "أخضر"; color_box = "#00ff00"
        elif wl >= 580e-9 and wl < 645e-9: region = "أصفر/برتقالي"; color_box = "#ffaa00"
        elif wl >= 645e-9 and wl <= 780e-9: region = "أحمر"; color_box = "#ff0000"
        elif wl > 780e-9: region = "IR"; color_box = "#440000"

        st.markdown(f"""
        <div class="card card-accent" style="border-left:4px solid {color_box};">
            <h4 style="color:#00e5ff;margin:0 0 12px 0;">📊 النتائج</h4>
            <p style="margin:6px 0;"><strong>النوع:</strong> {'انبعاث 🔴' if is_em else 'امتصاص 🔵'}</p>
            <div class="equation-box">dE = |{e_f:.3f} - ({e_i:.3f})| = <strong>{dE:.4f} eV</strong></div>
            <div style="background:rgba(0,229,255,0.05);border:1px solid rgba(0,229,255,0.1);border-radius:8px;padding:12px;margin:8px 0;direction:ltr;text-align:left;font-family:monospace;">
                <p style="margin:4px 0;color:#00e5ff;">E = {dE*EV_TO_J:.3e} J</p>
                <p style="margin:4px 0;color:#94a3b8;">lambda = {wl*1e9:.2f} nm</p>
                <p style="margin:4px 0;color:#94a3b8;">f = {freq:.3e} Hz</p>
            </div>
            <p><strong>المنطقة:</strong> <span style="background:{color_box};color:white;padding:2px 10px;border-radius:10px;font-size:0.85em;">{region}</span></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">📝 خطوات الحل</h3>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card">
        <p><strong>الخطوة 1:</strong> حساب طاقة كل مستوى:</p>
        <div class="equation-box">E_{calc_ni} = -13.6 / {calc_ni}^2 = {e_i:.3f} eV<br>E_{calc_nf} = -13.6 / {calc_nf}^2 = {e_f:.3f} eV</div>
        <p><strong>الخطوة 2:</strong> فرق الطاقة:</p>
        <div class="equation-box">dE = |{e_f:.3f} - ({e_i:.3f})| = {dE:.4f} eV</div>
        <p><strong>الخطوة 3:</strong> تحويل إلى جول:</p>
        <div class="equation-box">E = {dE:.4f} x 1.6x10^-19 = {dE*EV_TO_J:.3e} J</div>
        <p><strong>الخطوة 4:</strong> حساب التردد:</p>
        <div class="equation-box">f = E/h = {dE*EV_TO_J:.3e} / 6.63x10^-34 = {freq:.3e} Hz</div>
        <p><strong>الخطوة 5:</strong> حساب الطول الموجي:</p>
        <div class="equation-box">lambda = c/f = 3x10^8 / {freq:.3e} = {wl*1e9:.2f} nm</div>
    </div>
    """, unsafe_allow_html=True)
    t_code = "emit" if calc_nf < calc_ni else "absorption"
    components.html(energy_level_diagram(highlight_ni=calc_ni, highlight_nf=calc_nf, transition_type=t_code), height=470)

# ============================================================
# SECTION 4: ATOMIC SPECTRA
# ============================================================
elif section == "🌈 الأطياف الذرية":
    st.markdown('<h1 class="section-title">الأطياف الذرية | Atomic Spectra</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
        <h4 style="color:#ff6b35;margin:0 0 8px 0;">💡 من الحياة اليومية</h4>
        <p style="margin:0;">كل شخص له بصمة إصبع فريدة. كذلك كل عنصر كيميائي له "بصمة طيفية" فريدة — مجموعة خطوط ضوئية محددة.
        هذا المبدأ يُستخدم في تحديد عناصر النجوم البعيدة! قطرات المطر تعمل كمنشور فتُظهر قوس قزح (طيف متصل)،
        لكن ضوء عنصر واحد يعطي خطوطاً منفصلة فقط.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">📋 أنواع الأطياف</h3>', unsafe_allow_html=True)
    specs = [
        ("الطيف المتصل | Continuous Spectrum", "يحتوي جميع الأطوال الموجية بدون فواصل. أمثلة: ضوء الشمس، مصباح التنغستن."),
        ("طيف الانبعاث الخطي | Emission Line Spectrum", "خطوط مضيئة على خلفية سوداء. أمثلة: أنابيب النيون، أضواء المدينة."),
        ("طيف الامتصاص الخطي | Absorption Line Spectrum", "خطوط معتمة على خلفية مضيئة. أمثلة: خطوط فراونهوفر في طيف الشمس."),
    ]
    for t, d in specs:
        st.markdown(f'<div class="card"><h4 style="color:#00e5ff;margin:0 0 8px 0;">{t}</h4><p style="margin:0;color:#94a3b8;">{d}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box"><p style="margin:0;"><strong>⚠️ مهم:</strong> الخطوط المعتمة في طيف الامتصاص تقابل تماماً الخطوط المضيئة في طيف الانبعاث لنفس العنصر.</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-title">🎨 عارض الأطياف التفاعلي</h3>', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        spec_type = st.radio("نوع الطيف:", ["انبعاث خطي", "امتصاص خطي", "متصل"], key="sp_t")
    with sc2:
        spec_series = st.selectbox("المتتالية:", ["Balmer (مرئي)", "Lyman (UV)", "Paschen (IR)"], key="sp_s")
    tm = {"انبعاث خطي":"emission","امتصاص خطي":"absorption","متصل":"continuous"}
    sm = {"Balmer (مرئي)":"balmer","Lyman (UV)":"lyman","Paschen (IR)":"paschen"}
    components.html(spectrum_visualization(spec_type=tm[spec_type], series=sm[spec_series]), height=320)

    st.markdown('<h3 class="sub-title">📋 متتاليات طيف الهيدروجين</h3>', unsafe_allow_html=True)
    st.markdown("""
    <style>.st2{width:100%;border-collapse:collapse;font-size:0.85em;}
    .st2 th{background:rgba(0,229,255,0.1);color:#00e5ff;padding:8px;border:1px solid rgba(0,229,255,0.15);text-align:center;font-family:monospace;}
    .st2 td{padding:6px 8px;text-align:center;border:1px solid rgba(0,229,255,0.08);color:#e0e0e0;font-family:monospace;}</style>
    <table class="st2">
    <tr><th>المتتالية</th><th>n_f</th><th>n_i</th><th>المنطقة</th></tr>
    <tr><td style="color:#ff3366;">Lyman</td><td>1</td><td>2,3,4,...</td><td>UV</td></tr>
    <tr><td style="color:#00ff88;">Balmer</td><td>2</td><td>3,4,5,...</td><td>مرئي</td></tr>
    <tr><td style="color:#3388ff;">Paschen</td><td>3</td><td>4,5,6,...</td><td>IR</td></tr>
    <tr><td style="color:#ffcc00;">Brackett</td><td>4</td><td>5,6,7,...</td><td>IR</td></tr>
    <tr><td style="color:#cc66ff;">Pfund</td><td>5</td><td>6,7,8,...</td><td>IR</td></tr>
    </table>
    """, unsafe_allow_html=True)

# ============================================================
# SECTION 5: HYDROGEN SPECTRUM & RYDBERG
# ============================================================
elif section == "📐 الطيف الهيدروجيني":
    st.markdown('<h1 class="section-title">نموذج بور وطيف ذرة الهيدروجين</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
        <h4 style="color:#ff6b35;margin:0 0 8px 0;">💡 من الحياة اليومية</h4>
        <p style="margin:0;">لوحات الأرقام ليست عشوائية — كل رقم له معنى. كذلك خطوط طيف الهيدروجين تتبع معادلة دقيقة
        (معادلة ريدبرغ). نموذج بور نجح في اشتقاق هذه المعادلة من المبادئ الأولى!</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">📐 الاشتقاق الرياضي</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <p><strong>الخطوة 1:</strong> طاقة الفوتون:</p>
        <div class="equation-box">hf = 13.6e x |1/n_f^2 - 1/n_i^2|</div>
        <p style="margin-top:12px;"><strong>الخطوة 2:</strong> التعويض عن f = c/lambda والقسمة على hc:</p>
        <div class="equation-box">1/lambda = (13.6e)/(hc) x |1/n_f^2 - 1/n_i^2|</div>
        <p style="margin-top:12px;"><strong>الخطوة 3:</strong> ثابت ريدبرغ:</p>
        <div class="equation-box">R_H = 13.6e/(hc) = 1.097 x 10^7 m^-1</div>
        <p style="margin-top:12px;"><strong>الخطوة 4:</strong> المعادلة النهائية:</p>
        <div class="equation-box card-accent" style="font-size:1.5em;border-color:rgba(255,107,53,0.4);">1/lambda = R_H x |1/n_f^2 - 1/n_i^2|</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">✅ التحقق التجريبي — متتالية بالمر</h3>', unsafe_allow_html=True)
    ni_vals = [3,4,5,6]
    exp_vals = [656.2,486.1,434.0,410.1]
    rows = ""
    for ni, exp in zip(ni_vals, exp_vals):
        cal = calc_rydberg_wavelength(ni, 2)*1e9
        rows += f"<tr><td>{ni}</td><td>2</td><td>{cal:.1f}</td><td>{exp}</td><td>{abs(cal-exp):.1f}</td><td style='color:#00ff88;'>✓</td></tr>"
    st.markdown(f"""
    <style>.vt{{width:100%;border-collapse:collapse;}}
    .vt th{{background:rgba(0,229,255,0.1);color:#00e5ff;padding:10px;border:1px solid rgba(0,229,255,0.15);text-align:center;font-family:monospace;}}
    .vt td{{padding:8px;text-align:center;border:1px solid rgba(0,229,255,0.08);color:#e0e0e0;font-family:monospace;}}</style>
    <table class="vt"><tr><th>n_i</th><th>n_f</th><th>lambda محسوب</th><th>lambda تجريبي</th><th>الفرق</th><th>الحكم</th></tr>{rows}</table>
    <div class="card card-accent" style="margin-top:12px;"><p style="margin:0;"><strong style="color:#00ff88;">✅ النتيجة:</strong> القيم المحسوبة تقترب جداً من التجريبية مما يدل على صحة النموذج.
    <br><strong style="color:#ff6b35;">⚠️ حدود النموذج:</strong> فشل في تفسير أطياف الذرات عديدة الإلكترونات.</p></div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">🧮 حاسبة معادلة ريدبرغ</h3>', unsafe_allow_html=True)
    rc1, rc2 = st.columns(2)
    with rc1:
        r_ni = st.slider("n_i:", 1, 20, 3, key="ryd_ni")
    with rc2:
        r_nf = st.slider("n_f:", 1, 20, 2, key="ryd_nf")

    if r_ni != r_nf:
        wl_r = calc_rydberg_wavelength(r_ni, r_nf)
        de_r = calc_photon_energy(r_ni, r_nf)
        freq_r = calc_frequency(de_r)
        reg = "UV" if wl_r < 380e-9 else ("مرئي" if wl_r <= 780e-9 else "IR")
        clr_r = wavelength_to_rgb(wl_r * 1e9)
        sn = "balmer"
        if r_nf == 1: sn = "lyman"
        elif r_nf == 3: sn = "paschen"
        elif r_nf == 4: sn = "brackett"

        st.markdown(f"""
        <div class="card card-accent">
            <div class="equation-box">1/lambda = R_H x |1/{r_nf}^2 - 1/{r_ni}^2| = {1/wl_r:.3e} m^-1</div>
            <div style="display:flex;gap:20px;align-items:center;margin-top:12px;">
                <div style="background:{clr_r};width:40px;height:40px;border-radius:8px;border:2px solid rgba(255,255,255,0.3);"></div>
                <div style="font-family:monospace;">
                    <p style="margin:0;color:#00e5ff;">lambda = {wl_r*1e9:.2f} nm</p>
                    <p style="margin:0;color:#94a3b8;">f = {freq_r:.3e} Hz | dE = {de_r:.4f} eV | {reg}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        components.html(spectrum_visualization(spec_type="emission", series=sn), height=320)

# ============================================================
# SECTION 6: WAVE-PARTICLE DUALITY
# ============================================================
elif section == "🌊 الطبيعة المزدوجة":
    st.markdown('<h1 class="section-title">الطبيعة المزدوجة | Wave-Particle Duality</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
        <h4 style="color:#ff6b35;margin:0 0 8px 0;">💡 من الحياة اليومية</h4>
        <p style="margin:0;">فكّر في عملة معدنية: من جهة لها صورة (جسيم) ومن الجهة الأخرى كتابة (موجة).
        لا يمكنك رؤية الوجهين معاً! الضوء والإلكترونات هكذا: أحياناً تتصرف كموجات (حيود) وأحياناً كجسيمات (كهرضوئية).</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">🔬 الطبيعة المزدوجة للإشعاع</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
            <div style="background:rgba(0,229,255,0.05);border:1px solid rgba(0,229,255,0.2);border-radius:12px;padding:16px;">
                <h4 style="color:#00e5ff;margin:0 0 8px 0;">🌊 الطبيعة الموجية</h4>
                <ul style="padding-right:16px;color:#94a3b8;font-size:0.9em;margin:0;"><li>الحيود</li><li>التداخل</li><li>الاستقطاب</li></ul>
            </div>
            <div style="background:rgba(255,107,53,0.05);border:1px solid rgba(255,107,53,0.2);border-radius:12px;padding:16px;">
                <h4 style="color:#ff6b35;margin:0 0 8px 0;">⚛️ الطبيعة الجسيمية</h4>
                <ul style="padding-right:16px;color:#94a3b8;font-size:0.9em;margin:0;"><li>الظاهرة الكهرضوئية</li><li>تأثير كومبتون</li><li>الأطياف الذرية</li></ul>
            </div>
        </div>
        <p style="margin-top:12px;">زخم الفوتون كجسيم:</p>
        <div class="equation-box">p = h / lambda</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">🔮 نمط الحيود التفاعلي</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p style="margin:0;"><strong>ما هو الحيود؟</strong> عندما يمر موج عبر فتحة ضيقة ينحني وينتشر — هذا دليل على الطبيعة الموجية!
        تجربة <strong>دافسون وجيرمر (1927)</strong> أثبتت أن الإلكترونات تُظهر حيود عند سقوطها على بلورة نيكل.</p>
    </div>
    """, unsafe_allow_html=True)

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        diff_type = st.selectbox("نوع الجسيم:", ["ضوء", "إلكترونات"], key="diff_type")
    with dc2:
        diff_wl = st.slider("lambda (nm):", 100, 800, 500, key="diff_wl")
    with dc3:
        diff_slit = st.slider("عرض الشقة a (nm):", 100, 5000, 1000, key="diff_slit")

    dtc = "light" if diff_type == "ضوء" else "electron"
    if diff_type == "إلكترونات":
        diff_wl = 0.027
        diff_slit = 215
        st.info("للإلكترونات: lambda = 0.027 nm، المسافة بين ذرات النيكل = 0.215 nm")
    components.html(diffraction_pattern(wavelength_nm=diff_wl, slit_width_nm=diff_slit, particle_type=dtc), height=420)

    st.markdown("""
    <div class="card">
        <p><strong>معادلة الحيود:</strong></p>
        <div class="equation-box">I = I_0 x (sin(beta)/beta)^2   حيث   beta = pi*a*sin(theta)/lambda</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SECTION 7: DE BROGLIE HYPOTHESIS
# ============================================================
elif section == "🔬 فرضية دي بروي":
    st.markdown('<h1 class="section-title">فرضية دي بروي | De Broglie Hypothesis</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
        <h4 style="color:#ff6b35;margin:0 0 8px 0;">💡 من الحياة اليومية</h4>
        <p style="margin:0;">إذا رأيت موجة في البحر تعرف أنها موجة. وإذا رأيت كرة تعرف أنها جسم.
        لكن ماذا لو قلت لك أن الكرة التي ترميها لها "موجة مصاحبة" لا تستطيع رؤيتها لأنها صغيرة جداً؟
        هذا ما قالته فرضية دي بروي: كل جسم متحرك له موجة مصاحبة!</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">📝 نص الفرضية والاشتقاق</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card card-accent">
        <p style="font-size:1.05em;">"لكل جسم متحرك موجة مصاحبة:</p>
        <div class="equation-box" style="font-size:1.6em;margin:16px 0;">lambda = h / (mv)</div>
        <p><strong>الاشتقاق:</strong> من الطبيعة المزدوجة للضوء: p_photon = h/lambda.
        دي بروي عمّم: p = mv = h/lambda ← lambda = h/(mv)</p>
        <ul style="padding-right:20px;margin-top:12px;">
            <li>الموجات المصاحبة ليست موجات ميكانيكية أو كهرمغناطيسية.</li>
            <li>للأجسام الكبيرة: lambda صغير جداً ← لا يُلاحظ.</li>
            <li>للإلكترون: lambda يقارب المسافة بين الذرات ← يُلاحظ الحيود.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">🧮 حاسبة طول موجة دي بروي</h3>', unsafe_allow_html=True)
    db_preset = st.selectbox("اختر جسيماً:", [
        "إلكترون (v = 1.0 x 10^6 m/s)",
        "رصاصة (m = 50g, v = 400 m/s)",
        "كرة تنس (m = 60g, v = 6.5 m/s)",
        "بروتون (KE = 10 MeV)",
        "إلكترون مُسرَّع (dV = 2.7 V)",
        "قيمة مخصصة"
    ], key="db_preset")

    if db_preset == "إلكترون (v = 1.0 x 10^6 m/s)":
        db_mass, db_vel = ME, 1.0e6
    elif db_preset == "رصاصة (m = 50g, v = 400 m/s)":
        db_mass, db_vel = 50e-3, 400
    elif db_preset == "كرة تنس (m = 60g, v = 6.5 m/s)":
        db_mass, db_vel = 60e-3, 6.5
    elif db_preset == "بروتون (KE = 10 MeV)":
        mp = 1.67e-27
        db_mass = mp
        db_vel = (2 * 10e6 * EV_TO_J / mp) ** 0.5
    elif db_preset == "إلكترون مُسرَّع (dV = 2.7 V)":
        db_mass = ME
        db_vel = (2 * 2.7 * EV_TO_J / ME) ** 0.5
    else:
        dbc1, dbc2 = st.columns(2)
        with dbc1:
            db_mass = st.number_input("الكتلة m (kg):", 1e-35, 1e3, ME, format="%e", key="db_mass")
        with dbc2:
            db_vel = st.number_input("السرعة v (m/s):", 0.0, 1e10, 1e6, format="%e", key="db_vel")

    if db_vel > 0 and db_mass > 0:
        db_wl = calc_de_broglie_wavelength(db_mass, db_vel)
        db_p = db_mass * db_vel
        can_obs = db_wl > 1e-11

        wl_d = db_wl
        wl_u = "m"
        if db_wl < 1e-12: wl_d = db_wl*1e15; wl_u = "fm"
        elif db_wl < 1e-9: wl_d = db_wl*1e12; wl_u = "pm"
        elif db_wl < 1e-6: wl_d = db_wl*1e9; wl_u = "nm"
        elif db_wl < 1e-3: wl_d = db_wl*1e6; wl_u = "um"

        st.markdown(f"""
        <div class="card card-accent">
            <div class="equation-box" style="font-size:1.2em;">lambda = h/(mv) = {H_PLANCK:.2e}/({db_mass:.2e} x {db_vel:.2e}) = <strong>{db_wl:.3e} m</strong></div>
            <div style="background:rgba(0,229,255,0.05);border:1px solid rgba(0,229,255,0.1);border-radius:8px;padding:14px;margin:12px 0;direction:ltr;text-align:left;font-family:monospace;">
                <p style="margin:4px 0;color:#00e5ff;">lambda = {wl_d:.3f} {wl_u}</p>
                <p style="margin:4px 0;color:#94a3b8;">p = mv = {db_p:.3e} kg.m/s</p>
                <p style="margin:4px 0;color:#94a3b8;">m = {db_mass:.2e} kg</p>
                <p style="margin:4px 0;color:#94a3b8;">v = {db_vel:.2e} m/s</p>
            </div>
            <div style="padding:12px;border-radius:8px;background:{'rgba(0,255,136,0.1);border:1px solid rgba(0,255,136,0.3)' if can_obs else 'rgba(255,107,53,0.1);border:1px solid rgba(255,107,53,0.3)'};">
                <p style="margin:0;color:{'#00ff88' if can_obs else '#ff6b35'};">
                {'✅ يمكن ملاحظة الحيود' if can_obs else '❌ لا يمكن ملاحظة الحيود (lambda أصغر بكثير من المسافة بين الذرات)'}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        components.html(de_broglie_wave(mass_kg=db_mass, velocity=db_vel), height=320)

    el_wl = calc_de_broglie_wavelength(ME, 1e6)
    bul_wl = calc_de_broglie_wavelength(50e-3, 400)
    st.markdown(f"""
    <style>.ct3{{width:100%;border-collapse:collapse;}}
    .ct3 th{{background:rgba(0,229,255,0.1);color:#00e5ff;padding:10px;border:1px solid rgba(0,229,255,0.15);text-align:center;font-family:monospace;}}
    .ct3 td{{padding:8px;text-align:center;border:1px solid rgba(0,229,255,0.08);color:#e0e0e0;font-family:monospace;}}</style>
    <table class="ct3">
    <tr><th>الجسم</th><th>الكتلة</th><th>السرعة</th><th>lambda</th><th>المسافة بين الذرات</th><th>النتيجة</th></tr>
    <tr><td style="color:#00e5ff;">إلكترون</td><td>{ME:.2e} kg</td><td>1.0x10^6</td>
        <td style="color:#00ff88;"><strong>{el_wl*1e9:.3f} nm</strong></td><td>~0.3 nm</td>
        <td style="color:#00ff88;">✅ حيود ظاهر</td></tr>
    <tr><td style="color:#ff6b35;">رصاصة</td><td>5.0x10^-2 kg</td><td>400</td>
        <td style="color:#ff6b35;"><strong>{bul_wl:.2e} m</strong></td><td>~0.3 nm</td>
        <td style="color:#ff6b35;">❌ لا حيود</td></tr>
    </table>
    <div class="info-box" style="margin-top:12px;"><p style="margin:0;"><strong>الخلاصة:</strong> lambda الإلكترون يقارب المسافة بين الذرات فنرى الحيود. أما lambda الرصاصة أصغر بـ 10^26 مرة!</p></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="border-left:4px solid #22d3ee;">
        <h4 style="color:#22d3ee;margin:0 0 8px 0;">🔬 المجهر الإلكتروني</h4>
        <p style="margin:0;">لأن lambda الإلكترون المُسرَّع أقصر ~100 مرة من lambda الضوء المرئي، فإن المجهر الإلكتروني يميّز تفاصيل أصغر بـ ~100 مرة — مما يسمح برؤية الفيروسات بدقة عالية.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SECTION 8: X-RAYS
# ============================================================
elif section == "🏥 الأشعة السينية":
    st.markdown('<h1 class="section-title">الأشعة السينية | X-Rays</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
        <h4 style="color:#ff6b35;margin:0 0 8px 0;">💡 من الحياة اليومية</h4>
        <p style="margin:0;">عندما تذهب إلى طبيب الأسنان أو تُصاب بكسر، يطلب الطبيب "صورة أشعة سينية".
        تُسرَّع إلكترونات بجهد عالٍ وتُصطدم بمعدن، فتنبعث أشعة تخترق اللحم لكنها تتوقف عند العظم!</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">🔬 ما هي الأشعة السينية؟</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <p>اكتشفها العالم <strong>رونتغن (1895)</strong>: أشعة ذات طاقة كبيرة تنبعث عند اصطدام إلكترونات عالية الطاقة بسطح فلزّ.</p>
        <div class="equation-box">lambda_X-ray = 10^-11 — 10^-8 m</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">🔧 أنبوب الأشعة السينية</h3>', unsafe_allow_html=True)
    components.html(xray_tube_diagram(), height=370)
    st.markdown("""
    <div class="card">
        <ol style="padding-right:20px;">
            <li><strong style="color:#ff6b35;">المهبط (الفتيل):</strong> يُسخَّن فينبعث منه إلكترونات.</li>
            <li><strong style="color:#4488ff;">المصعد:</strong> مادة معدنية (تنگستن) تصطدم بها الإلكترونات.</li>
            <li><strong style="color:#ff3366;">فرق الجهد العالي:</strong> يُسرِّع الإلكترونات نحو المصعد.</li>
            <li><strong style="color:#ffcc00;">نافذة الخروج:</strong> تسمح للأشعة بالخروج.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">📊 طيف الأشعة السينية</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
            <div style="background:rgba(0,229,255,0.05);border:1px solid rgba(0,229,255,0.2);border-radius:12px;padding:16px;">
                <h4 style="color:#00e5ff;margin:0 0 8px 0;">الطيف المتصل (Bremsstrahlung)</h4>
                <p style="margin:0;color:#94a3b8;font-size:0.9em;">تباطؤ الإلكترونات وفقدان طاقة حركية.</p>
            </div>
            <div style="background:rgba(255,107,53,0.05);border:1px solid rgba(255,107,53,0.2);border-radius:12px;padding:16px;">
                <h4 style="color:#ff6b35;margin:0 0 8px 0;">الطيف الخطي (المميز)</h4>
                <p style="margin:0;color:#94a3b8;font-size:0.9em;">تحرير إلكترون داخلي وانتقال خارجي لملء الفراغ → خطوط Ka و Kb.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    xc1, xc2 = st.columns(2)
    with xc1:
        xv = st.slider("جهد التسريع dV (kV):", 10, 100, 50, key="xv_kv")
    with xc2:
        xz = st.selectbox("عنصر المصعد:", ["التنگستن (Z=74)", "النحاس (Z=29)", "الموليديدنوم (Z=42)"], key="xz_sel")
    xv_v = xv * 1000
    min_wl = H_PLANCK * C_LIGHT / (EV_TO_J * xv_v) * 1e9
    components.html(xray_spectrum(min_wl_nm=min_wl), height=370)
    st.markdown(f"""
    <div class="card card-accent">
        <div class="equation-box">lambda_min = hc/(e*dV) = {min_wl:.4f} nm</div>
        <p style="margin-top:8px;color:#94a3b8;">عند جهد {xv} kV، لا يمكن أن تكون الأشعة أقصر من {min_wl:.4f} nm.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">🏥 الاستخدامات الطبية</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card" style="border-left:4px solid #ff3366;">
        <p><strong>مبدأ التصوير:</strong> الأشعة تنفذ بنسب متفاوتة:</p>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
            <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:12px;text-align:center;">
                <p style="margin:0;color:#ffcc00;font-size:1.2em;">🦴 العظام</p>
                <p style="margin:4px 0 0 0;color:#94a3b8;font-size:0.85em;">امتصاص عالي ← بيضاء</p>
            </div>
            <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:12px;text-align:center;">
                <p style="margin:0;color:#ff6b35;font-size:1.2em;">🫁 الأنسجة الرخوة</p>
                <p style="margin:4px 0 0 0;color:#94a3b8;font-size:0.85em;">امتصاص منخفض ← داكنة</p>
            </div>
        </div>
        <p style="margin-top:12px;"><strong>التطبيقات:</strong> تصوير العظام، القفص الصدري، الأسنان، المقطعي (CT)، علاج الأورام.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SECTION 9: TECHNOLOGICAL APPLICATIONS
# ============================================================
elif section == "⚡ تطبيقات تكنولوجية":
    st.markdown('<h1 class="section-title">تطبيقات تكنولوجية | Applications</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
        <h4 style="color:#ff6b35;margin:0 0 8px 0;">💡 من الحياة اليومية</h4>
        <p style="margin:0;">عندما تقرأ رمز QR بموبايلك تستخدم الليزر. ولمبة الفلورسنت تعمل بنفس مبدأ انتقال الإلكترونات!
        كل هذه التكنولوجيات تعتمد على <strong>مبدأ واحد</strong>: انتقال الإلكترونات بين مستويات الطاقة.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">⚡ أشعة الليزر (LASER)</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <p><strong>LASER</strong> = Light Amplification by Stimulated Emission of Radiation</p>
        <ol style="padding-right:20px;">
            <li><strong>الضخ:</strong> تزويد الذرات بطاقة لرفع إلكتروناتها.</li>
            <li><strong>الانقلاب السكاني:</strong> عدد الذرات المثارة > المستقرة.</li>
            <li><strong>الانبعاث المحفَّز:</strong> فوتون يحفّز ذرة لإطلاق فوتون مماثل (نفس lambda واتجاه وطور).</li>
            <li><strong>التضخيم:</strong> المرآتان تعكسان الفوتونات فتتضاعف.</li>
            <li><strong>الخروج:</strong> المرآة الجزئية تسمح بالخروج كشعاع مركّز.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    components.html(laser_diagram(), height=320)

    st.markdown("""
    <div class="card" style="border-left:4px solid #ff0000;">
        <h4 style="color:#ff3333;margin:0 0 8px 0;">🎯 خصائص الليزر</h4>
        <ul style="padding-right:20px;margin:0;">
            <li><strong>اتجاهية عالية:</strong> شعاع ضيق جداً</li>
            <li><strong>أحادي اللون:</strong> كل الفوتونات لها نفس lambda</li>
            <li><strong>تماسك:</strong> كل الفوتونات في نفس الطور</li>
            <li><strong>شدة عالية:</strong> طاقة مركّزة في مساحة صغيرة</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">📋 تطبيقات الليزر</h3>', unsafe_allow_html=True)
    st.markdown("""
    <style>.at{{width:100%;border-collapse:collapse;}}
    .at th{{background:rgba(0,229,255,0.1);color:#00e5ff;padding:10px;border:1px solid rgba(0,229,255,0.15);text-align:center;}}
    .at td{{padding:8px 10px;border:1px solid rgba(0,229,255,0.08);color:#e0e0e0;vertical-align:top;}}</style>
    <table class="at">
    <tr><th style="width:100px;">المجال</th><th>التطبيق</th></tr>
    <tr><td style="color:#ff3333;font-weight:bold;">الطب</td><td>الليزك، علاج الأورام، جراحات دقيقة</td></tr>
    <tr><td style="color:#22d3ee;font-weight:bold;">الصناعة</td><td>قطع ونقش المعادن والخشب، اللحام الدقيق</td></tr>
    <tr><td style="color:#00ff88;font-weight:bold;">الاتصالات</td><td>نقل البيانات عبر الألياف الضوئية</td></tr>
    <tr><td style="color:#ffcc00;font-weight:bold;">الحياة اليومية</td><td>قارئات الباركود وQR، أقراص Blu-ray، مؤشرات الليزر</td></tr>
    <tr><td style="color:#cc66ff;font-weight:bold;">السيارات</td><td>أنظمة LiDAR للسيارات ذاتية القيادة</td></tr>
    <tr><td style="color:#ff6b35;font-weight:bold;">الهولوغرام</td><td>تصوير ثلاثي الأبعاد باستخدام التداخل</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-title">⚖️ مقارنة: الليزر vs الأشعة السينية</h3>', unsafe_allow_html=True)
    st.markdown("""
    <style>.cp{{width:100%;border-collapse:collapse;}}
    .cp th{{background:rgba(168,85,247,0.1);color:#a855f7;padding:10px;border:1px solid rgba(168,85,247,0.15);text-align:center;}}
    .cp td{{padding:8px 10px;border:1px solid rgba(168,85,247,0.08);color:#e0e0e0;}}</style>
    <table class="cp">
    <tr><th>الخاصية</th><th style="color:#ff3333;">الليزر</th><th style="color:#ffcc00;">الأشعة السينية</th></tr>
    <tr><td>الطول الموجي</td><td>400nm - 10um</td><td>0.01 - 10 nm</td></tr>
    <tr><td>مصدر الإنتاج</td><td>انبعاث محفَّز</td><td>تباطؤ إلكترونات + انتقالات داخلية</td></tr>
    <tr><td>الاتجاهية</td><td>شعاع ضيق جداً</td><td>منتشر</td></tr>
    <tr><td>الطيف</td><td>خط واحد</td><td>متصل + خطي</td></tr>
    <tr><td>المبدأ المشترك</td><td colspan="2" style="text-align:center;color:#00e5ff;">انتقال الإلكترونات بين مستويات الطاقة</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card card-accent" style="text-align:center;padding:30px;">
        <div style="font-size:2em;margin-bottom:12px;">🔗</div>
        <h3 style="color:#00e5ff;margin:0 0 12px 0;">الخيط المشترك</h3>
        <p style="max-width:600px;margin:0 auto;color:#94a3b8;">
        كل التطبيقات — الأشعة السينية، الليزر، أنابيب الفلورسنت، الميكروويف، شاشات الهاتف —
        تعتمد على <strong style="color:#ff6b35;">مبدأ واحد</strong>: انتقال الإلكترونات بين مستويات الطاقة المنفصلة في الذرات.
        فهم نموذج بور هو المفتاح لفهم كل هذه التكنولوجيات!</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:20px 0;">
    <p style="color:#94a3b8;font-size:0.9em;">التركيب الذري — فيزياء الصف الثاني عشر</p>
    <p style="color:#00e5ff;font-weight:700;font-size:1.05em;margin:8px 0 0 0;">إعداد: Israa Youssuf Samara</p>
</div>
""", unsafe_allow_html=True)
