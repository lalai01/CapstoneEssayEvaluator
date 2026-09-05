import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Stars } from '@react-three/drei';
import * as THREE from 'three';

// Floating abstract shapes – subtle and dark
function FloatingShapes() {
  const groupRef = useRef();
  const shapes = useMemo(() => {
    const count = 12;
    const items = [];
    for (let i = 0; i < count; i++) {
      const shape = Math.random() > 0.5 ? 'box' : 'sphere';
      const position = [
        (Math.random() - 0.5) * 14,
        (Math.random() - 0.5) * 10,
        (Math.random() - 0.5) * 12 - 6,
      ];
      // Dark professional colors: navy, slate, dark indigo
      const hue = 0.6 + Math.random() * 0.2; // blue range
      const color = new THREE.Color().setHSL(hue, 0.6, 0.25);
      const scale = Math.random() * 0.6 + 0.3;
      const speed = Math.random() * 0.3 + 0.1;
      items.push({ shape, position, color, scale, speed });
    }
    return items;
  }, []);

  useFrame(({ clock }) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = clock.getElapsedTime() * 0.03;
    }
  });

  return (
    <group ref={groupRef}>
      {shapes.map((item, i) => (
        <mesh key={i} position={item.position} scale={item.scale}>
          {item.shape === 'box' ? (
            <boxGeometry args={[1, 1, 1]} />
          ) : (
            <sphereGeometry args={[0.6, 16, 16]} />
          )}
          <meshStandardMaterial
            color={item.color}
            roughness={0.5}
            metalness={0.2}
            emissive={item.color}
            emissiveIntensity={0.15}
            transparent
            opacity={0.5}
          />
        </mesh>
      ))}
    </group>
  );
}

// Particle field – subtle stars
function ParticleField() {
  const count = 1500;
  const positions = useMemo(() => {
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 50;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 30;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 40 - 15;
    }
    return pos;
  }, []);

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial
        color="#6b8cae"
        size={0.04}
        transparent
        opacity={0.3}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

// Main abstract element – dark blue torus knot
function AnimatedKnot() {
  const meshRef = useRef();
  useFrame(({ clock }) => {
    if (meshRef.current) {
      meshRef.current.rotation.x = clock.getElapsedTime() * 0.08;
      meshRef.current.rotation.y = clock.getElapsedTime() * 0.12;
    }
  });
  return (
    <mesh ref={meshRef} position={[0, -0.5, -4]}>
      <torusKnotGeometry args={[1.8, 0.45, 128, 16, 3, 4]} />
      <meshStandardMaterial
        color="#1e3a5f"
        emissive="#0f2a4a"
        emissiveIntensity={0.3}
        roughness={0.4}
        metalness={0.6}
        wireframe={false}
        transparent
        opacity={0.75}
      />
    </mesh>
  );
}

export default function ThreeDBackground() {
  return (
    <div className="fixed inset-0 -z-20 pointer-events-none bg-gradient-to-br from-gray-900 via-slate-900 to-blue-950">
      <Canvas
        camera={{ position: [0, 0, 8], fov: 60 }}
        style={{ background: 'transparent' }}
        gl={{ alpha: true, antialias: true }}
      >
        <ambientLight intensity={0.3} />
        <pointLight position={[5, 8, 5]} intensity={0.5} />
        <pointLight position={[-5, -3, -5]} color="#2a4a6a" intensity={0.3} />
        <fog attach="fog" args={['#0a121c', 18, 30]} />

        <AnimatedKnot />
        <FloatingShapes />
        <ParticleField />
        <Stars radius={120} depth={60} count={2500} factor={5} saturation={0.2} fade speed={0.3} />
      </Canvas>
    </div>
  );
}