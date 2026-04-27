let sequence = [];
let recording = false;
let currentGesture = '';
let dataset = [];

let model = null;
let labels = [];

let predictionHistory = [];
const historySize = 10;

let lastSpokenLabel = '';
let lastSpokenTime = 0;

const SEQUENCE_LENGTH = 20;
const RECORD_TIME = 3000;

let countdown = 0;
let isCountingDown = false;

function setStatus(type, mainText, subText=""){

const box = document.getElementById("statusBox");
const main = document.getElementById("statusMain");
const sub = document.getElementById("statusSub");
const icon = box.querySelector(".status-icon");

box.className = "status " + type;
main.innerText = mainText;
sub.innerText = subText;

if(type === "success") icon.innerText = "✔️";
else if(type === "warning") icon.innerText = "⏳";
else if(type === "error") icon.innerText = "❗";
else icon.innerText = "ℹ️";
}

window.addEventListener("DOMContentLoaded", () => {

const videoElement = document.querySelector('.input_video');
const canvasElement = document.querySelector('.output_canvas');
const canvasCtx = canvasElement.getContext('2d');

canvasElement.width = 640;
canvasElement.height = 480;

function speak(text){
const msg = new SpeechSynthesisUtterance(text);
msg.lang = 'sv-SE';
msg.rate = 0.9;
speechSynthesis.cancel();
speechSynthesis.speak(msg);
}

function onResults(results){

canvasCtx.save();
canvasCtx.clearRect(0,0,canvasElement.width,canvasElement.height);
canvasCtx.drawImage(results.image,0,0,canvasElement.width,canvasElement.height);

if(results.multiHandLandmarks){

for(const landmarks of results.multiHandLandmarks){

drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,{color:'#00FF00'});
drawLandmarks(canvasCtx, landmarks,{color:'#FF0000'});

const frame = [];
landmarks.forEach(p=> frame.push(p.x,p.y,p.z));

// recording
if(recording){
sequence.push(frame);
}

// prediction
if(model){

sequence.push(frame);

if(sequence.length > SEQUENCE_LENGTH){
sequence.shift();
}

if(sequence.length === SEQUENCE_LENGTH){

const flat = [];
sequence.forEach(f => flat.push(...f));

const input = tf.tensor2d([flat]);
const prediction = model.predict(input);

prediction.array().then(scores=>{

const maxIndex = scores[0].indexOf(Math.max(...scores[0]));
const confidence = Math.max(...scores[0]);
const percent = Math.round(confidence * 100);

const label = labels[maxIndex];

// stabilisering
predictionHistory.push(label);
if(predictionHistory.length > historySize){
predictionHistory.shift();
}

const counts = {};
predictionHistory.forEach(l=>{
counts[l] = (counts[l] || 0) + 1;
});

let stableLabel = Object.keys(counts).reduce((a,b)=>
counts[a] > counts[b] ? a : b
);

// färg baserat på säkerhet
let color;

if(percent > 75) color = "lime";
else if(percent > 50) color = "yellow";
else color = "red";

// visa text + %
canvasCtx.font="32px Arial";
canvasCtx.fillStyle=color;
canvasCtx.fillText(stableLabel + " (" + percent + "%)",20,50);

// prata
const SPEAK_THRESHOLD = 75;

if(
  percent >= SPEAK_THRESHOLD &&
  stableLabel === label && // viktigt!
  (
    stableLabel !== lastSpokenLabel ||
    Date.now() - lastSpokenTime > 3000
  )
){
  speak(stableLabel);
  lastSpokenLabel = stableLabel;
  lastSpokenTime = Date.now();
}

});

}

}

}

}

// countdown overlay
if(isCountingDown){
canvasCtx.font = "80px Arial";
canvasCtx.fillStyle = "red";
canvasCtx.textAlign = "center";
canvasCtx.fillText(countdown, canvasElement.width/2, canvasElement.height/2);
}

canvasCtx.restore();
}

// MediaPipe
const hands = new Hands({
locateFile:(file)=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
maxNumHands:1,
modelComplexity:1,
minDetectionConfidence:0.7,
minTrackingConfidence:0.7
});

hands.onResults(onResults);

const camera = new Camera(videoElement,{
onFrame: async()=>{ await hands.send({image:videoElement}); },
width:640,
height:480
});

camera.start();

setStatus("success","Redo","Kamera aktiv");

// 🎤 RECORD
window.startRecording = function(){

const name = prompt("Namn på tecken:");

if(!name){
setStatus("error","Ingen text angiven");
return;
}

currentGesture = name;
sequence = [];

let count = 3;
countdown = count;
isCountingDown = true;

setStatus("warning","Gör dig redo...");

const interval = setInterval(()=>{

count--;

if(count > 0){
countdown = count;
}else{
clearInterval(interval);

isCountingDown = false;
recording = true;

setStatus("warning","Spelar in...","Lär verktyget ett nytt tecken");

setTimeout(()=>{

recording = false;

if(sequence.length === 0){
setStatus("error","Ingen data inspelad");
return;
}

if(sequence.length >= SEQUENCE_LENGTH){
sequence = sequence.slice(0, SEQUENCE_LENGTH);
}else{
while(sequence.length < SEQUENCE_LENGTH){
sequence.push(sequence[sequence.length - 1]);
}
}

dataset.push({ label: currentGesture, sequence: sequence });

sequence = [];

setStatus("success","Sekvens sparad!", currentGesture);

}, RECORD_TIME);

}

},1000);

};

// 💾 DOWNLOAD
window.downloadData = function(){

if(dataset.length === 0){
setStatus("error","Ingen data att spara");
return;
}

const dataStr="data:text/json;charset=utf-8,"+encodeURIComponent(JSON.stringify(dataset));
const link=document.createElement('a');
link.href=dataStr;
link.download="gesture_dataset.json";
link.click();

setStatus("success","Data sparad");
};

// 🤖 TRAIN
window.trainModel = async function(){

const fileInput = document.getElementById("datasetFile");

if(!fileInput.files.length){
setStatus("error","Ladda upp dataset först");
return;
}

setStatus("warning","Verktyget lär sig...");

const allData = [];

for(const file of fileInput.files){
const text = await file.text();
const data = JSON.parse(text);
allData.push(...data);
}

const xs=[], ys=[];
labels=[];

allData.forEach(item=>{
const flat=[];
item.sequence.forEach(f=>flat.push(...f));
xs.push(flat);

if(!labels.includes(item.label)) labels.push(item.label);
ys.push(labels.indexOf(item.label));
});

if(labels.length < 2){
setStatus("error","Minst två tecken behövs");
return;
}

const xsTensor=tf.tensor2d(xs);
const ysTensor=tf.oneHot(tf.tensor1d(ys,'int32'),labels.length);

model=tf.sequential();

model.add(tf.layers.dense({inputShape:[63*SEQUENCE_LENGTH],units:128,activation:'relu'}));
model.add(tf.layers.dense({units:64,activation:'relu'}));
model.add(tf.layers.dense({units:labels.length,activation:'softmax'}));

model.compile({optimizer:'adam',loss:'categoricalCrossentropy',metrics:['accuracy']});

await model.fit(xsTensor,ysTensor,{epochs:50,shuffle:true});

setStatus("success","Verktyget är redo!", labels.join(", "));
};

// upload feedback
document.getElementById("datasetFile").addEventListener("change", function(){
if(this.files.length > 0){
setStatus("success","Filer uppladdade", this.files.length + " filer");
}
});

});