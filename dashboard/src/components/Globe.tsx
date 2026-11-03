import { useEffect, useRef } from "react";
import * as THREE from "three";

// Realistic Earth: NASA Blue Marble colour, topology bump map, ocean specular map, night lights
// on the dark side, a cloud layer and a soft atmospheric rim glow. The satellite and its orbit
// are decorative and not a real spacecraft position.
const ATMOSPHERE_VERTEX = `
varying vec3 vNormal;
void main() {
  vNormal = normalize(normalMatrix * normal);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
const ATMOSPHERE_FRAGMENT = `
varying vec3 vNormal;
void main() {
  float intensity = pow(0.72 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 3.2);
  gl_FragColor = vec4(0.22, 0.75, 1.0, 1.0) * intensity;
}`;

export default function Globe({ className = "" }: { className?: string }) {
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
    camera.position.set(0, 0, 4.1);

    const loader = new THREE.TextureLoader();
    const tex = (name: string, srgb = false) => {
      const t = loader.load(`/textures/${name}`);
      if (srgb) t.colorSpace = THREE.SRGBColorSpace;
      t.anisotropy = renderer.capabilities.getMaxAnisotropy();
      return t;
    };

    const earthGroup = new THREE.Group();
    earthGroup.rotation.z = THREE.MathUtils.degToRad(-23.4);
    scene.add(earthGroup);

    const sunDir = new THREE.Vector3(-1.2, 0.35, 0.9).normalize();
    const earthMaterial = new THREE.MeshPhongMaterial({
      map: tex("earth-blue-marble.jpg", true),
      bumpMap: tex("earth-topology.png"),
      bumpScale: 3,
      specularMap: tex("earth-water.png"),
      specular: new THREE.Color("#5b7fa8"),
      shininess: 14,
      emissiveMap: tex("earth-night.jpg", true),
      emissive: new THREE.Color("#ffd28a"),
      emissiveIntensity: 0.0,
    });
    // Show city lights only on the night side
    earthMaterial.onBeforeCompile = (shader) => {
      shader.uniforms.sunDir = { value: sunDir };
      shader.fragmentShader = shader.fragmentShader
        .replace("#include <common>", "#include <common>\nuniform vec3 sunDir;")
        .replace(
          "#include <emissivemap_fragment>",
          `#ifdef USE_EMISSIVEMAP
            vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
            float nightSide = smoothstep(0.15, -0.25, dot(normalize(vNormal), normalize((viewMatrix * vec4(sunDir, 0.0)).xyz)));
            totalEmissiveRadiance = emissiveColor.rgb * vec3(1.0, 0.82, 0.55) * nightSide * 1.25;
          #endif`,
        );
    };
    const earth = new THREE.Mesh(new THREE.SphereGeometry(1, 96, 96), earthMaterial);
    earthGroup.add(earth);

    const clouds = new THREE.Mesh(
      new THREE.SphereGeometry(1.012, 96, 96),
      new THREE.MeshPhongMaterial({ map: tex("clouds.png", true), transparent: true, opacity: 0.38, depthWrite: false }),
    );
    earthGroup.add(clouds);

    const atmosphere = new THREE.Mesh(
      new THREE.SphereGeometry(1.16, 64, 64),
      new THREE.ShaderMaterial({
        vertexShader: ATMOSPHERE_VERTEX,
        fragmentShader: ATMOSPHERE_FRAGMENT,
        blending: THREE.AdditiveBlending,
        side: THREE.BackSide,
        transparent: true,
        depthWrite: false,
      }),
    );
    scene.add(atmosphere);

    // Decorative orbit and satellite (illustrative only)
    const orbitGroup = new THREE.Group();
    orbitGroup.rotation.set(THREE.MathUtils.degToRad(68), THREE.MathUtils.degToRad(12), 0);
    scene.add(orbitGroup);
    const orbitRadius = 1.42;
    const orbit = new THREE.Mesh(
      new THREE.TorusGeometry(orbitRadius, 0.0018, 8, 256),
      new THREE.MeshBasicMaterial({ color: "#38e8ff", transparent: true, opacity: 0.35 }),
    );
    orbitGroup.add(orbit);
    const sat = new THREE.Group();
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.035, 0.05), new THREE.MeshStandardMaterial({ color: "#d9e6ff", metalness: 0.7, roughness: 0.3 }));
    const panelMat = new THREE.MeshStandardMaterial({ color: "#1e4fa8", metalness: 0.4, roughness: 0.35, emissive: "#0b2a66", emissiveIntensity: 0.6 });
    const panelL = new THREE.Mesh(new THREE.BoxGeometry(0.11, 0.002, 0.035), panelMat);
    const panelR = panelL.clone();
    panelL.position.x = -0.075;
    panelR.position.x = 0.075;
    const beacon = new THREE.Mesh(new THREE.SphereGeometry(0.012, 16, 16), new THREE.MeshBasicMaterial({ color: "#38e8ff" }));
    beacon.position.y = 0.03;
    sat.add(body, panelL, panelR, beacon);
    orbitGroup.add(sat);

    const stars = new THREE.BufferGeometry();
    const positions = new Float32Array(1800 * 3);
    for (let i = 0; i < 1800; i++) {
      const r = 30 + Math.random() * 20;
      const th = Math.random() * Math.PI * 2;
      const ph = Math.acos(2 * Math.random() - 1);
      positions.set([r * Math.sin(ph) * Math.cos(th), r * Math.sin(ph) * Math.sin(th), r * Math.cos(ph)], i * 3);
    }
    stars.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    scene.add(new THREE.Points(stars, new THREE.PointsMaterial({ color: "#cfe6ff", size: 0.06, transparent: true, opacity: 0.8 })));

    const sun = new THREE.DirectionalLight("#ffffff", 2.6);
    sun.position.copy(sunDir).multiplyScalar(10);
    scene.add(sun, new THREE.AmbientLight("#2a3d66", 0.35));

    const resize = () => {
      const { clientWidth: w, clientHeight: h } = el;
      renderer.setSize(w, h, false);
      renderer.domElement.style.width = "100%";
      renderer.domElement.style.height = "100%";
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    resize();

    let pointerX = 0;
    let pointerY = 0;
    const onPointer = (e: PointerEvent) => {
      const rect = el.getBoundingClientRect();
      pointerX = ((e.clientX - rect.left) / rect.width - 0.5) * 2;
      pointerY = ((e.clientY - rect.top) / rect.height - 0.5) * 2;
    };
    el.addEventListener("pointermove", onPointer);

    let frame = 0;
    let angle = 0;
    const clock = new THREE.Clock();
    const animate = () => {
      const dt = Math.min(clock.getDelta(), 0.05);
      const speed = reduceMotion ? 0 : 1;
      earth.rotation.y += 0.045 * dt * speed;
      clouds.rotation.y += 0.06 * dt * speed;
      angle += 0.32 * dt * speed;
      sat.position.set(Math.cos(angle) * orbitRadius, Math.sin(angle) * orbitRadius, 0);
      sat.rotation.z = angle;
      scene.rotation.y += (pointerX * 0.12 - scene.rotation.y) * 0.04;
      scene.rotation.x += (pointerY * 0.06 - scene.rotation.x) * 0.04;
      renderer.render(scene, camera);
      frame = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      el.removeEventListener("pointermove", onPointer);
      scene.traverse((obj) => {
        const mesh = obj as THREE.Mesh;
        mesh.geometry?.dispose();
        const mats = Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : [];
        mats.forEach((m) => {
          Object.values(m).forEach((v) => v instanceof THREE.Texture && v.dispose());
          m.dispose();
        });
      });
      renderer.dispose();
      el.removeChild(renderer.domElement);
    };
  }, []);

  return <div ref={host} className={className} aria-label="Rotating 3D Earth" role="img" />;
}
