import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { ContactShadows, Environment, Lightformer, RoundedBox } from '@react-three/drei';
import * as THREE from 'three';

// A friendly service robot reading an essay on a tablet: glossy shell, glass
// visor, jointed arms, hover base. It breathes, blinks, glances down at the
// essay and turns its head toward the mouse pointer.

const SHELL = { color: '#f4f6fb', roughness: 0.22, metalness: 0.05, clearcoat: 1, clearcoatRoughness: 0.08 };
const JOINT = { color: '#2a3140', roughness: 0.35, metalness: 0.7 };
const VISOR = { color: '#0a0f1c', roughness: 0.04, metalness: 0.3, clearcoat: 1, clearcoatRoughness: 0.02 };
const GLOW = '#38bdf8';

// A capsule stretched between two points (upper arm, forearm...).
function Limb({ from, to, radius }) {
  const { position, quaternion, length } = useMemo(() => {
    const a = new THREE.Vector3(...from);
    const b = new THREE.Vector3(...to);
    const dir = b.clone().sub(a);
    return {
      position: a.clone().add(b).multiplyScalar(0.5),
      quaternion: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize()),
      length: dir.length(),
    };
  }, [from, to]);
  return (
    <mesh position={position} quaternion={quaternion} castShadow>
      <capsuleGeometry args={[radius, Math.max(0.01, length - radius * 2), 8, 16]} />
      <meshPhysicalMaterial {...SHELL} />
    </mesh>
  );
}

function Joint({ position, radius = 0.1 }) {
  return (
    <mesh position={position} castShadow>
      <sphereGeometry args={[radius, 24, 24]} />
      <meshStandardMaterial {...JOINT} />
    </mesh>
  );
}

// Arm geometry: shoulder -> elbow -> hand on the tablet's edge.
const ARMS = [-1, 1].map((side) => ({
  shoulder: [0.6 * side, -0.02, 0],
  elbow: [0.66 * side, -0.5, 0.18],
  hand: [0.4 * side, -0.6, 0.62],
}));

function Robot() {
  const body = useRef();
  const head = useRef();
  const eyes = useRef();
  const chest = useRef();
  const beam = useRef();
  const ring = useRef();
  const look = useRef({ x: 0, y: 0 });

  useFrame(({ clock, pointer }) => {
    const t = clock.getElapsedTime();

    // Breathing / hovering.
    if (body.current) {
      body.current.position.y = Math.sin(t * 1.4) * 0.04;
      body.current.rotation.y = Math.sin(t * 0.5) * 0.06;
    }

    // Head: follow the pointer, and glance down at the essay every few seconds.
    const glance = Math.sin(t * 0.7) > 0.55 ? 0.35 : 0;
    const targetX = THREE.MathUtils.clamp(-pointer.y * 0.35, -0.3, 0.3) + glance;
    const targetY = THREE.MathUtils.clamp(pointer.x * 0.6, -0.55, 0.55);
    look.current.x = THREE.MathUtils.lerp(look.current.x, targetX, 0.06);
    look.current.y = THREE.MathUtils.lerp(look.current.y, targetY, 0.06);
    if (head.current) {
      head.current.rotation.x = look.current.x;
      head.current.rotation.y = look.current.y;
      head.current.rotation.z = Math.sin(t * 0.9) * 0.03;
    }

    // Blink: a quick squash of the eyes every ~3.8 s.
    if (eyes.current) {
      const phase = t % 3.8;
      eyes.current.scale.y = phase < 0.12 ? Math.max(0.08, Math.abs(phase - 0.06) / 0.06) : 1;
    }

    // Chest light breathes; the scan beam sweeps the page.
    if (chest.current) chest.current.emissiveIntensity = 1.4 + Math.sin(t * 2.2) * 0.6;
    if (beam.current) beam.current.position.y = Math.sin(t * 1.3) * 0.2;
    if (ring.current) ring.current.material.opacity = 0.45 + Math.sin(t * 1.4) * 0.15;
  });

  return (
    <group position={[0, 0.08, 0]}>
      <group ref={body}>
        {/* Hover base */}
        <mesh position={[0, -1.1, 0]} castShadow>
          <cylinderGeometry args={[0.36, 0.24, 0.38, 48]} />
          <meshPhysicalMaterial {...SHELL} />
        </mesh>
        <mesh position={[0, -1.31, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[0.22, 0.025, 16, 64]} />
          <meshStandardMaterial color={GLOW} emissive={GLOW} emissiveIntensity={2} toneMapped={false} />
        </mesh>

        {/* Torso */}
        <RoundedBox args={[1.05, 1.0, 0.68]} radius={0.24} smoothness={6} position={[0, -0.42, 0]} castShadow>
          <meshPhysicalMaterial {...SHELL} />
        </RoundedBox>
        <mesh position={[0, -0.84, 0]}>
          <cylinderGeometry args={[0.4, 0.38, 0.12, 48]} />
          <meshStandardMaterial {...JOINT} />
        </mesh>
        {/* Chest light */}
        <mesh position={[0, -0.22, 0.345]}>
          <circleGeometry args={[0.075, 32]} />
          <meshStandardMaterial ref={chest} color={GLOW} emissive={GLOW} emissiveIntensity={1.6} toneMapped={false} />
        </mesh>
        <mesh position={[0, -0.22, 0.34]}>
          <torusGeometry args={[0.1, 0.014, 12, 48]} />
          <meshStandardMaterial {...JOINT} />
        </mesh>

        {/* Neck */}
        <mesh position={[0, 0.16, 0]}>
          <cylinderGeometry args={[0.12, 0.15, 0.2, 32]} />
          <meshStandardMaterial {...JOINT} />
        </mesh>

        {/* Head */}
        <group ref={head} position={[0, 0.58, 0]}>
          <RoundedBox args={[0.96, 0.72, 0.74]} radius={0.3} smoothness={8} castShadow>
            <meshPhysicalMaterial {...SHELL} />
          </RoundedBox>
          {/* Glass visor */}
          <RoundedBox args={[0.78, 0.44, 0.1]} radius={0.16} smoothness={8} position={[0, 0.0, 0.33]}>
            <meshPhysicalMaterial {...VISOR} />
          </RoundedBox>
          {/* Eyes and smile, glowing behind the glass */}
          <group ref={eyes} position={[0, 0.05, 0.39]}>
            {[-0.17, 0.17].map((x) => (
              <RoundedBox key={x} args={[0.15, 0.11, 0.02]} radius={0.05} smoothness={4} position={[x, 0, 0]}>
                <meshStandardMaterial color={GLOW} emissive={GLOW} emissiveIntensity={2.4} toneMapped={false} />
              </RoundedBox>
            ))}
          </group>
          <mesh position={[0, -0.09, 0.39]} rotation={[0, 0, Math.PI]}>
            <torusGeometry args={[0.07, 0.012, 8, 32, Math.PI]} />
            <meshStandardMaterial color={GLOW} emissive={GLOW} emissiveIntensity={1.8} toneMapped={false} />
          </mesh>
          {/* Ears */}
          {[-1, 1].map((side) => (
            <group key={side} position={[0.49 * side, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
              <mesh>
                <cylinderGeometry args={[0.12, 0.12, 0.07, 32]} />
                <meshStandardMaterial {...JOINT} />
              </mesh>
              <mesh position={[0, 0.04 * side, 0]} rotation={[Math.PI / 2, 0, 0]}>
                <torusGeometry args={[0.075, 0.012, 12, 32]} />
                <meshStandardMaterial color={GLOW} emissive={GLOW} emissiveIntensity={1.5} toneMapped={false} />
              </mesh>
            </group>
          ))}
          {/* Antenna */}
          <mesh position={[0.22, 0.43, 0]}>
            <cylinderGeometry args={[0.012, 0.012, 0.16, 12]} />
            <meshStandardMaterial {...JOINT} />
          </mesh>
          <mesh position={[0.22, 0.53, 0]}>
            <sphereGeometry args={[0.035, 16, 16]} />
            <meshStandardMaterial color="#f97316" emissive="#f97316" emissiveIntensity={2} toneMapped={false} />
          </mesh>
        </group>

        {/* Arms */}
        {ARMS.map(({ shoulder, elbow, hand }) => (
          <group key={shoulder[0]}>
            <Joint position={shoulder} radius={0.15} />
            <Limb from={shoulder} to={elbow} radius={0.09} />
            <Joint position={elbow} radius={0.09} />
            <Limb from={elbow} to={hand} radius={0.08} />
            <Joint position={hand} radius={0.085} />
          </group>
        ))}

        {/* Tablet with the essay being scanned */}
        <group position={[0, -0.58, 0.7]} rotation={[-0.35, 0, 0]}>
          <RoundedBox args={[0.82, 0.6, 0.04]} radius={0.04} smoothness={4} castShadow>
            <meshStandardMaterial {...JOINT} />
          </RoundedBox>
          <mesh position={[0, 0, 0.022]}>
            <planeGeometry args={[0.72, 0.5]} />
            <meshStandardMaterial color="#f8fafc" roughness={0.6} />
          </mesh>
          {[0.17, 0.09, 0.01, -0.07, -0.15].map((y, i) => (
            <mesh key={y} position={[i === 4 ? -0.1 : 0, y, 0.024]}>
              <planeGeometry args={[i === 4 ? 0.4 : 0.6, 0.025]} />
              <meshStandardMaterial color="#94a3b8" />
            </mesh>
          ))}
          <mesh position={[0.22, -0.16, 0.025]}>
            <planeGeometry args={[0.1, 0.06]} />
            <meshStandardMaterial color="#22c55e" emissive="#22c55e" emissiveIntensity={0.6} />
          </mesh>
          <mesh ref={beam} position={[0, 0, 0.03]}>
            <planeGeometry args={[0.74, 0.02]} />
            <meshStandardMaterial color={GLOW} emissive={GLOW} emissiveIntensity={3} transparent opacity={0.85} toneMapped={false} />
          </mesh>
        </group>
      </group>

      {/* Hover glow on the floor */}
      <mesh ref={ring} position={[0, -1.47, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.18, 0.42, 48]} />
        <meshBasicMaterial color={GLOW} transparent opacity={0.5} toneMapped={false} />
      </mesh>
      <ContactShadows position={[0, -1.48, 0]} opacity={0.45} scale={3.5} blur={2.6} far={2.2} />
    </group>
  );
}

// Studio lighting baked into an environment map (no downloads), so the
// glossy shell and visor get soft, realistic reflections.
function Studio() {
  return (
    <Environment resolution={256}>
      <Lightformer form="rect" intensity={3} position={[0, 4, 3]} scale={[6, 2, 1]} />
      <Lightformer form="rect" intensity={1.5} position={[-4, 1, 2]} rotation-y={Math.PI / 2} scale={[4, 3, 1]} />
      <Lightformer form="rect" intensity={1.2} color="#bae6fd" position={[4, 1, -2]} rotation-y={-Math.PI / 2} scale={[4, 3, 1]} />
      <Lightformer form="circle" intensity={2} position={[0, 0, -5]} scale={3} />
    </Environment>
  );
}

// The robot spans about x -1.0…1.0 and y -1.5…1.2; the camera frames it at
// any canvas size.
export default function RobotAssistant() {
  return (
    <div className="w-full h-full min-w-[8rem] min-h-[8rem] relative">
      <Canvas
        camera={{ position: [0, 0.05, 4.3], fov: 42 }}
        dpr={[1, 1.75]}
        style={{ background: 'transparent', width: '100%', height: '100%' }}
        gl={{ alpha: true, antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}
      >
        <ambientLight intensity={0.25} />
        <directionalLight position={[3, 4, 4]} intensity={1.6} />
        <directionalLight position={[-3, 2, -2]} intensity={0.6} color="#bae6fd" />
        <Studio />
        <Robot />
      </Canvas>
    </div>
  );
}
