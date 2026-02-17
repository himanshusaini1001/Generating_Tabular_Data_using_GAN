const generateBtn = document.getElementById("generate_btn");
const fastaBtn = document.getElementById("download_fasta");
const csvBtn = document.getElementById("download_csv");
const output = document.getElementById("output");
const gcLabel = document.getElementById("gc_label");
const gcProgress = document.getElementById("gc_progress");
const baseChartCanvas = document.getElementById("baseChart");
const seqDropdown = document.getElementById("sequence_dropdown");
const seqChartCanvas = document.getElementById("seqChart");
const comparisonChartCanvas = document.getElementById("comparisonChart");
const qualityScoreElement = document.getElementById("quality_score");

let latestFasta = "";
let latestSequences = [];
let baseChart, seqChart, comparisonChart;

// ---------- Base chart ----------
function updateChart(a,t,g,c){
    if(baseChart) baseChart.destroy();
    baseChart = new Chart(baseChartCanvas, {
        type: 'bar',
        data: {
            labels: ['A', 'T', 'G', 'C'],
            datasets: [{
                label: 'Base Composition %',
                data: [a, t, g, c],
                backgroundColor: ['#FF4D6D','#4DFFB8','#4D6DFF','#FFD24D'],
                borderColor: ['#FF1A3C','#1AFF9B','#1A3CFF','#FFC700'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {legend: {display:false}},
            scales: {
                y: {beginAtZero:true, max:100, ticks:{color:'#fff', font:{size:12}}},
                x: {ticks:{color:'#fff', font:{size:12}}}
            }
        }
    });
}

// ---------- Per-sequence chart ----------
function updateSeqChart(sequence){
    let counts={A:0,T:0,G:0,C:0};
    sequence.split('').forEach(b => { if(counts[b]!==undefined) counts[b]++; });
    const total = sequence.length;
    const percents = [
        counts.A/total*100,
        counts.T/total*100,
        counts.G/total*100,
        counts.C/total*100
    ];

    if(seqChart) seqChart.destroy();
    seqChart = new Chart(seqChartCanvas, {
        type:'bar',
        data:{
            labels:['A','T','G','C'],
            datasets:[{
                label:'Composition % (This Seq)',
                data: percents,
                backgroundColor:['#FF4D6D','#4DFFB8','#4D6DFF','#FFD24D'],
                borderColor:['#FF1A3C','#1AFF9B','#1A3CFF','#FFC700'],
                borderWidth:2
            }]
        },
        options:{
            responsive: true,
            maintainAspectRatio: false,
            plugins:{legend:{display:false}},
            scales:{
                y:{beginAtZero:true, max:100, ticks:{color:'#fff', font:{size:10}}},
                x:{ticks:{color:'#fff', font:{size:10}}}
            }
        }
    });
}

// ---------- Comparison Chart ----------
function updateComparisonChart(syntheticData) {
    if(comparisonChart) comparisonChart.destroy();
    
    // Mock original data for comparison (you can replace this with real data)
    const originalData = {
        A: 25.2, T: 24.8, G: 25.1, C: 24.9
    };
    
    comparisonChart = new Chart(comparisonChartCanvas, {
        type: 'bar',
        data: {
            labels: ['A', 'T', 'G', 'C'],
            datasets: [
                {
                    label: 'Original',
                    data: [originalData.A, originalData.T, originalData.G, originalData.C],
                    backgroundColor: 'rgba(255, 77, 109, 0.6)',
                    borderColor: '#FF4D6D',
                    borderWidth: 1
                },
                {
                    label: 'Synthetic',
                    data: [syntheticData.A, syntheticData.T, syntheticData.G, syntheticData.C],
                    backgroundColor: 'rgba(77, 255, 184, 0.6)',
                    borderColor: '#4DFFB8',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#fff',
                        font: { size: 10 }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 30,
                    ticks: { color: '#fff', font: { size: 8 } }
                },
                x: {
                    ticks: { color: '#fff', font: { size: 8 } }
                }
            }
        }
    });
}

// ---------- Calculate Quality Score ----------
function calculateQualityScore(syntheticData) {
    // Mock original data
    const originalData = { A: 25.2, T: 24.8, G: 25.1, C: 24.9 };
    
    let totalDiff = 0;
    ['A', 'T', 'G', 'C'].forEach(base => {
        totalDiff += Math.abs(syntheticData[base] - originalData[base]);
    });
    
    // Convert difference to quality score (lower difference = higher quality)
    const qualityScore = Math.max(0, 100 - (totalDiff * 2));
    return Math.round(qualityScore);
}

// ---------- 3D DNA ----------
let scene, camera, renderer, helixGroup;

function initDNA3D(){
    const container = document.getElementById("dna3d");
    container.innerHTML = ""; // clear old canvas

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(50, container.clientWidth/container.clientHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    helixGroup = new THREE.Group();
    scene.add(helixGroup);

    camera.position.z = 25;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(20,20,20);
    scene.add(pointLight);

    animateDNA();
}

function animateDNA(){
    requestAnimationFrame(animateDNA);
    if(helixGroup) helixGroup.rotation.y += 0.01; // simple rotation
    renderer.render(scene, camera);
}

// ---------- Draw DNA helix ----------
function drawDNA(sequence){
    // Clear previous
    for(let i=helixGroup.children.length-1;i>=0;i--){
        const obj = helixGroup.children[i];
        helixGroup.remove(obj);
        if(obj.geometry) obj.geometry.dispose();
        if(obj.material) obj.material.dispose();
    }

    const n = sequence.length;
    const radius = 1.2, helixHeight = 0.25;
    const geometry = new THREE.CylinderGeometry(0.05,0.05,helixHeight,16);
    const colors = {A:0xFF4D6D, T:0x4DFFB8, G:0x4D6DFF, C:0xFFD24D};
    const compBases = {A:'T', T:'A', G:'C', C:'G'};

    for(let i=0;i<n;i++){
        const angle = i*0.55;
        const y = i*helixHeight - (n*helixHeight)/2;

        const base1 = sequence[i] in colors ? sequence[i] : 'A';
        const cyl1 = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
            color: colors[base1], emissive: colors[base1], emissiveIntensity:0.3
        }));
        cyl1.position.set(Math.cos(angle)*radius, y, Math.sin(angle)*radius);
        helixGroup.add(cyl1);

        const base2 = compBases[sequence[i]] || 'T';
        const cyl2 = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
            color: colors[base2], emissive: colors[base2], emissiveIntensity:0.3
        }));
        cyl2.position.set(Math.cos(angle+Math.PI)*radius, y, Math.sin(angle+Math.PI)*radius);
        helixGroup.add(cyl2);
    }

    helixGroup.rotation.y = 0;
}

// ---------- Generate sequences ----------
generateBtn.onclick = async function(){
    const n = document.getElementById("num_sequences").value;
    const formData = new URLSearchParams();
    formData.append("num_sequences", n);

    try{
        const res = await fetch("/generate", {method:"POST", body: formData});
        const data = await res.json();

        output.textContent = "";
        latestFasta = data.fasta;
        latestSequences = data.sequences;

        // Dropdown
        seqDropdown.innerHTML = "";
        data.sequences.forEach((seq,i)=>{
            const opt = document.createElement("option");
            opt.value = i;
            opt.textContent = `Seq${i+1}`;
            seqDropdown.appendChild(opt);
        });

        // Output sequences & GC
        const minGC=30, maxGC=60;
        data.sequences.forEach((seq,i)=>{
            const gc = seq.split('').filter(b=>b=='G'||b=='C').length/seq.length*100;
            if(gc<minGC||gc>maxGC){
                output.textContent += `>Seq${i+1} [GC:${gc.toFixed(1)}%] << Outside target\n${seq}\n`;
            } else {
                output.textContent += `>Seq${i+1} [GC:${gc.toFixed(1)}%]\n${seq}\n`;
            }
        });

        gcLabel.textContent = "GC Content: "+data.gc_content.toFixed(2)+"%";
        gcProgress.value = data.gc_content;

        let totalBases=0, counts={A:0,T:0,G:0,C:0};
        data.sequences.forEach(seq=>{
            totalBases += seq.length;
            seq.split('').forEach(b=>{ if(counts[b]!==undefined) counts[b]++; });
        });
        
        const syntheticData = {
            A: counts.A/totalBases*100,
            T: counts.T/totalBases*100,
            G: counts.G/totalBases*100,
            C: counts.C/totalBases*100
        };
        
        updateChart(syntheticData.A, syntheticData.T, syntheticData.G, syntheticData.C);
        updateComparisonChart(syntheticData);
        
        // Update quality score
        const qualityScore = calculateQualityScore(syntheticData);
        qualityScoreElement.textContent = qualityScore + '%';

        if(data.sequences.length>0){
            drawDNA(data.sequences[0]);
            updateSeqChart(data.sequences[0]);
        }

        // Show hidden sections
        document.querySelectorAll(".hidden").forEach(el => el.classList.add("show"));
    } catch(err){
        alert("Error generating: " + err);
    }
}

// ---------- Dropdown change ----------
seqDropdown.onchange = function(){
    const index = parseInt(seqDropdown.value);
    if(latestSequences[index]){
        drawDNA(latestSequences[index]);
        updateSeqChart(latestSequences[index]);
    }
}

// ---------- Downloads ----------
fastaBtn.onclick = function(){
    if(!latestFasta) return alert("Generate sequences first!");
    const blob = new Blob([latestFasta], {type:"text/plain"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "generated_sequences.fasta";
    a.click();
}

csvBtn.onclick = function(){
    if(!latestSequences.length) return alert("Generate sequences first!");
    let csv = "ID,Sequence\n";
    latestSequences.forEach((seq,i)=>{ csv += `Seq${i+1},${seq}\n`; });
    const blob = new Blob([csv], {type:"text/csv"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "generated_sequences.csv";
    a.click();
}

// ---------- Init 3D DNA on page load ----------
window.addEventListener("DOMContentLoaded", initDNA3D);
