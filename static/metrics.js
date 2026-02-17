const uploadBtn = document.getElementById("upload_btn");
const dnaFileInput = document.getElementById("dna_file");
const gcLabel = document.getElementById("gc_label");
const gcProgress = document.getElementById("gc_progress");
const baseChartCanvas = document.getElementById("baseChart");
const seqDropdown = document.getElementById("sequence_dropdown");

let latestSequences = [];
let baseChart;

function computeBaseStats(seq){
    const counts = {A:0,T:0,G:0,C:0};
    for(const ch of seq){ if(counts[ch]!==undefined) counts[ch]++; }
    const total = seq.length || 0;
    const gc = total>0 ? ((counts.G+counts.C)/total*100) : 0;
    return {counts,total,gc};
}

// ---------- Dinucleotide Heatmap (4x4) ----------
function computeDinucMatrix(seq){
    const bases=['A','T','G','C'];
    const matrix = Array.from({length:4},()=>Array(4).fill(0));
    if(!seq || seq.length<2) return {matrix,max:0};
    let max=0;
    for(let i=0;i<seq.length-1;i++){
        const a = bases.indexOf(seq[i]);
        const b = bases.indexOf(seq[i+1]);
        if(a===-1||b===-1) continue;
        matrix[a][b]++;
        if(matrix[a][b]>max) max = matrix[a][b];
    }
    return {matrix,max};
}
function renderDinucHeatmap(seq){
    const container = document.getElementById('diHeatmap');
    if(!container) return;
    const {matrix,max} = computeDinucMatrix(seq);
    const bases=['A','T','G','C'];
    const colorFor = (v)=>{
        if(max===0) return 'rgba(0,0,0,0)';
        const t = v/max;
        const r = Math.round(30 + 180*t);
        const g = Math.round(80 + 100*t);
        const b = Math.round(120 + 40*t);
        return `rgba(${r},${g},${b},0.9)`;
    };
    let html = '<table style="border-collapse:collapse;width:100%;height:100%;table-layout:fixed">';
    html += '<thead><tr><th style="color:#fff"></th>' + bases.map(b=>`<th style="color:#fff">${b}</th>`).join('') + '</tr></thead><tbody>';
    for(let i=0;i<4;i++){
        html += `<tr><th style="color:#fff">${bases[i]}</th>`;
        for(let j=0;j<4;j++){
            const v = matrix[i][j];
            html += `<td style="border:1px solid rgba(255,255,255,0.1);text-align:center;color:#fff;background:${colorFor(v)}">${v}</td>`;
        }
        html += '</tr>';
    }
    html += '</tbody></table>';
    container.innerHTML = html;
}

// ---------- Base Chart ----------
function updateChart(a,t,g,c){
    if(baseChart) baseChart.destroy();
    baseChart=new Chart(baseChartCanvas,{
        type:'bar',
        data:{
            labels:['A','T','G','C'],
            datasets:[{
                label:'Base Composition %',
                data:[a,t,g,c],
                backgroundColor:['#FF4D6D','#4DFFB8','#4D6DFF','#FFD24D'],
                borderColor:['#FF1A3C','#1AFF9B','#1A3CFF','#FFC700'],
                borderWidth:2
            }]
        },
        options:{
            responsive:true,
            maintainAspectRatio:false,
            plugins:{legend:{display:false}},
            scales:{
                y:{beginAtZero:true,max:100,ticks:{color:'#fff',font:{size:14}}},
                x:{ticks:{color:'#fff',font:{size:14}}}
            }
        }
    });
}

// ---------- 3D DNA ----------
let scene, camera, renderer, helixGroup;
function initDNA3D(){
    const container=document.getElementById("dna3d");
    scene=new THREE.Scene();
    camera=new THREE.PerspectiveCamera(75, container.clientWidth/container.clientHeight,0.1,1000);
    renderer=new THREE.WebGLRenderer({antialias:true, alpha:true});
    renderer.setSize(container.clientWidth,container.clientHeight);
    container.appendChild(renderer.domElement);

    helixGroup=new THREE.Group();
    scene.add(helixGroup);

    camera.position.z=15;
    scene.add(new THREE.AmbientLight(0xffffff,0.6));
    const pointLight=new THREE.PointLight(0xffffff,1);
    pointLight.position.set(20,20,20);
    scene.add(pointLight);

    animateDNA();
}
function animateDNA(){
    requestAnimationFrame(animateDNA);
    if(helixGroup) helixGroup.rotation.y+=0.01;
    renderer.render(scene,camera);
}
function drawDNA(sequence){
    for(let i=helixGroup.children.length-1;i>=0;i--){
        const obj=helixGroup.children[i];
        helixGroup.remove(obj);
        if(obj.geometry)obj.geometry.dispose();
        if(obj.material)obj.material.dispose();
    }
    const n=sequence.length;
    const radius=1.2, helixHeight=0.25;
    const geometry=new THREE.CylinderGeometry(0.05,0.05,helixHeight,16);
    const colors={A:0xFF4D6D,T:0x4DFFB8,G:0x4D6DFF,C:0xFFD24D};
    const compBases={A:'T',T:'A',G:'C',C:'G'};

    for(let i=0;i<n;i++){
        const angle=i*0.55;
        const y=i*helixHeight - (n*helixHeight)/2;
        const base1=sequence[i] in colors ? sequence[i] : 'A';
        const cyl1=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:colors[base1],emissive:colors[base1],emissiveIntensity:0.3}));
        cyl1.position.set(Math.cos(angle)*radius, y, Math.sin(angle)*radius);
        helixGroup.add(cyl1);

        const base2=compBases[sequence[i]] || 'T';
        const cyl2=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:colors[base2],emissive:colors[base2],emissiveIntensity:0.3}));
        cyl2.position.set(Math.cos(angle+Math.PI)*radius, y, Math.sin(angle+Math.PI)*radius);
        helixGroup.add(cyl2);
    }
    helixGroup.rotation.y=0;
}
initDNA3D();

// ---------- Functional & Motif Metrics ----------
const TFBS_MOTIFS = ["TATA","CAAT","GC"];
function countTFBS(seq){
    let count = 0;
    TFBS_MOTIFS.forEach(motif=>{
        count += (seq.match(new RegExp(motif,"g"))||[]).length;
    });
    return count;
}
function countRepeats(seq){
    let repeats=0;
    for(let len=1;len<=6;len++){
        for(let i=0;i<=seq.length-len*3;i++){
            const sub = seq.substr(i,len);
            if(seq.substr(i,len*3) === sub.repeat(3)) repeats++;
        }
    }
    return repeats;
}
function kmerDistribution(seq,k=3){
    const kmers={};
    for(let i=0;i<=seq.length-k;i++){
        const sub = seq.substr(i,k);
        kmers[sub] = (kmers[sub]||0)+1;
    }
    return Object.entries(kmers).sort((a,b)=>b[1]-a[1]).slice(0,5).map(e=>`${e[0]}:${e[1]}`).join(", ");
}
function countConservedRegions(seq,window=6){
    let conserved=0;
    for(let i=0;i<=seq.length-window;i++){
        const sub=seq.substr(i,window);
        if(seq.indexOf(sub,i+1)!==-1) conserved++;
    }
    return conserved;
}
function countCpG(seq){
    return (seq.match(/CG/g)||[]).length;
}

// ---------- Upload & Analyze ----------
uploadBtn.onclick = async () => {
    const file = dnaFileInput.files[0];
    if(!file) return alert("Select a FASTA or CSV file first!");
    const text = await file.text();

    latestSequences = [];
    if(file.name.endsWith(".csv")){
        text.split("\n").forEach(line => {
            const seq = line.split(",")[1]?.trim();
            if(seq) latestSequences.push(seq.toUpperCase());
        });
    } else {
        let seq = "";
        text.split("\n").forEach(line=>{
            if(line.startsWith(">")){
                if(seq) latestSequences.push(seq.toUpperCase());
                seq="";
            } else seq += line.trim();
        });
        if(seq) latestSequences.push(seq.toUpperCase());
    }

    // Compute per-sequence composition & GC for first sequence
    const firstSeq = latestSequences[0] || "";
    const {counts,total,gc} = computeBaseStats(firstSeq);
    gcLabel.textContent = "GC Content: "+gc.toFixed(2)+"%";
    gcProgress.value = gc;
    if(total>0){
        updateChart(counts.A/total*100, counts.T/total*100, counts.G/total*100, counts.C/total*100);
    } else {
        updateChart(0,0,0,0);
    }

    // Show dropdown & draw first sequence
    seqDropdown.innerHTML="";
    latestSequences.forEach((seq,i)=>{
        const opt=document.createElement("option");
        opt.value=i;
        opt.textContent=`Seq${i+1}`;
        seqDropdown.appendChild(opt);
    });
    if(latestSequences.length>0) drawDNA(latestSequences[0]);
    renderDinucHeatmap(firstSeq);

    // Compute functional metrics
    const seq = latestSequences[0] || "";
    document.getElementById("cpg_count_label").textContent = `CpG Islands Count: ${countCpG(seq)}`;
    document.getElementById("tfbs_label").textContent = `TFBS Count: ${countTFBS(seq)}`;
    document.getElementById("repeats_label").textContent = `Repeat Elements Count: ${countRepeats(seq)}`;
    document.getElementById("kmer_label").textContent = `k-mer Distribution (Top 5): ${kmerDistribution(seq)}`;
    document.getElementById("conserved_label").textContent = `Conserved Regions Count: ${countConservedRegions(seq)}`;

    // Show hidden sections
    document.querySelectorAll(".hidden").forEach(el=>el.classList.add("show"));
}

seqDropdown.onchange = ()=>{
    const index=parseInt(seqDropdown.value);
    const seq = latestSequences[index] || "";
    if(seq) drawDNA(seq);

    // Update heatmap
    renderDinucHeatmap(seq);

    // Update per-sequence base chart and GC
    const {counts,total,gc} = computeBaseStats(seq);
    gcLabel.textContent = "GC Content: "+gc.toFixed(2)+"%";
    gcProgress.value = gc;
    if(total>0){
        updateChart(counts.A/total*100, counts.T/total*100, counts.G/total*100, counts.C/total*100);
    } else {
        updateChart(0,0,0,0);
    }

    document.getElementById("cpg_count_label").textContent = `CpG Islands Count: ${countCpG(seq)}`;
    document.getElementById("tfbs_label").textContent = `TFBS Count: ${countTFBS(seq)}`;
    document.getElementById("repeats_label").textContent = `Repeat Elements Count: ${countRepeats(seq)}`;
    document.getElementById("kmer_label").textContent = `k-mer Distribution (Top 5): ${kmerDistribution(seq)}`;
    document.getElementById("conserved_label").textContent = `Conserved Regions Count: ${countConservedRegions(seq)}`;
}
