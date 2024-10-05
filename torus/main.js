import './style.css';
import * as THREE from 'three';

// loading screen
window.onload=()=>{
  const loading=document.querySelector('.loading');
  loading.style.display='none';
  const temp1=document.querySelector('#btn');
  temp1.style.display='block';
  const temp2=document.querySelector('#hint');
  temp2.style.display='block';
}

// Setup
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({
  canvas: document.querySelector('#bg'),
});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
camera.position.setZ(10);
camera.position.setX(0);
camera.position.setY(0);
// Torus ( try them everyone once) variants
// const geometry = new THREE.TorusGeometry(10, 2, 16, 100,360);
// const geometry = new THREE.TorusGeometry(10, 2, 16, 100,220);
// const geometry = new THREE.TorusGeometry(10, 2, 16, 100,250);
// const geometry = new THREE.TorusGeometry(10, 2, 16, 100,180);
// const geometry = new THREE.TorusGeometry(10, 2, 16, 100,50);
const geometry = new THREE.TorusGeometry(10, 2, 16, 100,390);
// const geometry = new THREE.TorusGeometry(10, 2, 16, 100,450);
// const geometry = new THREE.TorusGeometry(10, 2, 16, 100,520);

const material = new THREE.MeshNormalMaterial();
const torus = new THREE.Mesh(geometry, material);
scene.add(torus);
let XrotationSpeed=0.01;
let YrotationSpeed=0.01;
let ZrotationSpeed=0.01;
let Zzoom=1;
// Lights
const pointLight = new THREE.PointLight(0xffffff);
pointLight.position.set(0, 0, 15);
scene.add(pointLight);
// Animation Loop
function animate() {
  requestAnimationFrame(animate);
  torus.rotation.x+=XrotationSpeed;
  torus.rotation.y+=YrotationSpeed;
  torus.rotation.z+=ZrotationSpeed;
  renderer.render(scene,camera);
}
animate();

// menu button
let btn=false;
const button=document.querySelector('#btn')
button.addEventListener('click',()=>{
  const menu = document.getElementById('menu');
  const hint=document.getElementById('hint');
  if(btn==false){
    menu.style.display = menu.style.display === 'block' ? 'none' : 'block';
    menu.style.position='fixed';
    menu.style.zIndex=98;
    menu.style.top='60vh';
    menu.style.left=0;
    hint.textContent='(click👆 to close menu)';
    btn=true;
  }else{
    menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
    hint.style.display='none';
    btn=false;
}
});

// menu sliders
const XspeedRange = document.getElementById('XspeedRange');
const XspeedValue = document.getElementById('XspeedValue');
XspeedRange.addEventListener('input', function (event) {
  XrotationSpeed = parseFloat(event.target.value);
  XspeedValue.textContent = XrotationSpeed.toFixed(3); // Display the current speed value
});
const YspeedRange = document.getElementById('YspeedRange');
const YspeedValue = document.getElementById('YspeedValue');
YspeedRange.addEventListener('input', function (event) {
  YrotationSpeed = parseFloat(event.target.value);
  YspeedValue.textContent = YrotationSpeed.toFixed(3); // Display the current speed value
});
const ZspeedRange = document.getElementById('ZspeedRange');
const ZspeedValue = document.getElementById('ZspeedValue');
ZspeedRange.addEventListener('input', function (event) {
  ZrotationSpeed = parseFloat(event.target.value);
  ZspeedValue.textContent = ZrotationSpeed.toFixed(3); // Display the current speed value
});
const ZzoomRange = document.getElementById('ZzoomRange');
const ZzoomValue = document.getElementById('ZzoomValue');
ZzoomRange.addEventListener('input', function (event) {
  Zzoom = parseInt(event.target.value);
  camera.position.setZ(Zzoom);
  ZzoomValue.textContent = `${Zzoom.toFixed(0)}%`; // Display the current speed value
});

// window resizing 
window.addEventListener('resize', function () {
  const width = window.innerWidth;
  const height = window.innerHeight;
  renderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
});