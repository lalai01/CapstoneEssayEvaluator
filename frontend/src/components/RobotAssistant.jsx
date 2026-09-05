import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float } from '@react-three/drei';
import * as THREE from 'three';

// Robot model with document scanning beam
function Robot({ isHovered }) {
  const groupRef = useRef();
  const headRef = useRef();
  const leftArmRef = useRef();
  const rightArmRef = useRef();
  const paperRef = useRef();
  const scanLineRef = useRef();
  const antennaMatRef = useRef();
  const leftEyeMatRef = useRef();
  const rightEyeMatRef = useRef();

  // Scanning animation state
  const [scanPos, setScanPos] = useState(-0.5); // from -0.5 to 0.5

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();

    // Gentle idle animation
    if (groupRef.current) {
      groupRef.current.position.y = Math.sin(t * 1.5) * 0.03;
    }
    if (headRef.current) {
      headRef.current.rotation.z = Math.sin(t * 2) * 0.1;
      headRef.current.rotation.x = Math.sin(t * 1.8) * 0.05;
    }
    if (leftArmRef.current) {
      leftArmRef.current.rotation.z = Math.sin(t * 2.5) * 0.2 - 0.3;
    }
    if (rightArmRef.current) {
      rightArmRef.current.rotation.z = Math.sin(t * 2.5 + 2) * 0.15 + 0.2;
    }

    // Paper floating
    if (paperRef.current) {
      paperRef.current.rotation.y += 0.01;
      paperRef.current.position.y = Math.sin(t * 1.2) * 0.05;
    }

    // SCANNING BEAM: moves up and down the paper continuously
    const speed = 1.2;
    const newScanPos = Math.sin(t * speed) * 0.55; // range -0.55 to 0.55
    if (scanLineRef.current) {
      scanLineRef.current.position.y = newScanPos;
      // Pulse intensity based on speed
      const intensity = 0.6 + Math.sin(t * 8) * 0.4;
      if (scanLineRef.current.material) {
        scanLineRef.current.material.emissiveIntensity = intensity;
      }
    }

    // Pulsing antenna and eyes during scan (makes it feel alive)
    const pulse = 0.4 + Math.sin(t * 6) * 0.3;
    if (antennaMatRef.current) {
      antennaMatRef.current.emissiveIntensity = pulse;
    }
    if (leftEyeMatRef.current && rightEyeMatRef.current) {
      const eyeIntensity = 0.7 + Math.sin(t * 5) * 0.4;
      leftEyeMatRef.current.emissiveIntensity = eyeIntensity;
      rightEyeMatRef.current.emissiveIntensity = eyeIntensity;
    }
  });

  return (
    <group ref={groupRef} position={[0, -0.5, 0]}>
      {/* Body */}
      <mesh position={[0, 0.2, 0]} castShadow receiveShadow>
        <boxGeometry args={[1.2, 1.4, 0.8]} />
        <meshStandardMaterial color="#4a90e2" roughness={0.4} metalness={0.3} />
      </mesh>

      {/* Head */}
      <group ref={headRef} position={[0, 1.1, 0]}>
        <mesh castShadow receiveShadow>
          <boxGeometry args={[0.9, 0.8, 0.7]} />
          <meshStandardMaterial color="#6b9cf0" roughness={0.3} metalness={0.2} />
        </mesh>
        {/* Eyes (with pulsing emissive) */}
        <mesh position={[-0.25, 0.15, 0.36]} castShadow>
          <sphereGeometry args={[0.12, 16, 16]} />
          <meshStandardMaterial
            ref={leftEyeMatRef}
            color="#ffffff"
            emissive="#aaccff"
            emissiveIntensity={0.8}
          />
        </mesh>
        <mesh position={[0.25, 0.15, 0.36]} castShadow>
          <sphereGeometry args={[0.12, 16, 16]} />
          <meshStandardMaterial
            ref={rightEyeMatRef}
            color="#ffffff"
            emissive="#aaccff"
            emissiveIntensity={0.8}
          />
        </mesh>
        {/* Pupils */}
        <mesh position={[-0.25, 0.1, 0.45]} castShadow>
          <sphereGeometry args={[0.05, 8, 8]} />
          <meshStandardMaterial color="#1a1a2e" />
        </mesh>
        <mesh position={[0.25, 0.1, 0.45]} castShadow>
          <sphereGeometry args={[0.05, 8, 8]} />
          <meshStandardMaterial color="#1a1a2e" />
        </mesh>
        {/* Antenna with pulsing material */}
        <mesh position={[0, 0.55, 0]} castShadow>
          <cylinderGeometry args={[0.05, 0.05, 0.4]} />
          <meshStandardMaterial color="#ffaa00" emissive="#ff8800" emissiveIntensity={0.5} />
        </mesh>
        <mesh position={[0, 0.8, 0]} castShadow>
          <sphereGeometry args={[0.1, 8, 8]} />
          <meshStandardMaterial
            ref={antennaMatRef}
            color="#ff5500"
            emissive="#ff3300"
            emissiveIntensity={0.8}
          />
        </mesh>
      </group>

      {/* Left Arm */}
      <group ref={leftArmRef} position={[-0.8, 0.5, 0]}>
        <mesh castShadow receiveShadow>
          <boxGeometry args={[0.3, 0.9, 0.3]} />
          <meshStandardMaterial color="#3a7bd5" roughness={0.5} />
        </mesh>
        <mesh position={[-0.15, -0.5, 0]} castShadow>
          <sphereGeometry args={[0.18, 8, 8]} />
          <meshStandardMaterial color="#4a90e2" />
        </mesh>
      </group>

      {/* Right Arm */}
      <group ref={rightArmRef} position={[0.8, 0.5, 0]}>
        <mesh castShadow receiveShadow>
          <boxGeometry args={[0.3, 0.9, 0.3]} />
          <meshStandardMaterial color="#3a7bd5" roughness={0.5} />
        </mesh>
        <mesh position={[0.15, -0.5, 0]} castShadow>
          <sphereGeometry args={[0.18, 8, 8]} />
          <meshStandardMaterial color="#4a90e2" />
        </mesh>
      </group>

      {/* Main Document with Scanning Beam */}
      <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
        <group ref={paperRef} position={[0.8, 0.3, 0.8]} rotation={[0.2, 0.5, 0.1]}>
          {/* Paper */}
          <mesh castShadow receiveShadow>
            <boxGeometry args={[0.8, 1.0, 0.02]} />
            <meshStandardMaterial color="#ffffff" roughness={0.7} emissive="#e2e8f0" emissiveIntensity={0.2} />
          </mesh>
          {/* Text lines on paper */}
          <mesh position={[0, 0.2, 0.02]} rotation={[0, 0, 0]}>
            <boxGeometry args={[0.6, 0.03, 0.01]} />
            <meshStandardMaterial color="#a0aec0" />
          </mesh>
          <mesh position={[0, 0.05, 0.02]} rotation={[0, 0, 0]}>
            <boxGeometry args={[0.5, 0.03, 0.01]} />
            <meshStandardMaterial color="#a0aec0" />
          </mesh>
          <mesh position={[0, -0.1, 0.02]} rotation={[0, 0, 0]}>
            <boxGeometry args={[0.55, 0.03, 0.01]} />
            <meshStandardMaterial color="#a0aec0" />
          </mesh>
          <mesh position={[0, -0.25, 0.02]} rotation={[0, 0, 0]}>
            <boxGeometry args={[0.4, 0.03, 0.01]} />
            <meshStandardMaterial color="#a0aec0" />
          </mesh>
          {/* Red check mark */}
          <mesh position={[0.2, -0.35, 0.03]} rotation={[0, 0, 0.3]}>
            <boxGeometry args={[0.1, 0.03, 0.01]} />
            <meshStandardMaterial color="#ef4444" />
          </mesh>
          <mesh position={[0.15, -0.4, 0.03]} rotation={[0, 0, -0.4]}>
            <boxGeometry args={[0.1, 0.03, 0.01]} />
            <meshStandardMaterial color="#ef4444" />
          </mesh>

          {/* SCANNING BEAM – a glowing line that moves vertically over the document */}
          <mesh ref={scanLineRef} position={[0, -0.5, 0.03]} scale={[0.75, 0.03, 0.01]}>
            <boxGeometry args={[1, 1, 1]} />
            <meshStandardMaterial color="#00ffff" emissive="#00ffff" emissiveIntensity={0.8} transparent opacity={0.9} />
          </mesh>

          {/* Glowing border around paper to emphasize scanning */}
          <mesh position={[0, 0, 0.04]} scale={[0.85, 1.05, 1]}>
            <boxGeometry args={[1, 1, 0.002]} />
            <meshStandardMaterial color="#00aaff" emissive="#0088ff" emissiveIntensity={0.3} transparent opacity={0.4} wireframe />
          </mesh>
        </group>
      </Float>

      {/* Second smaller paper */}
      <Float speed={1.5} rotationIntensity={0.3} floatIntensity={0.4}>
        <mesh position={[-0.6, 0.5, 0.6]} rotation={[0.1, -0.3, 0.2]}>
          <boxGeometry args={[0.5, 0.7, 0.02]} />
          <meshStandardMaterial color="#fef9c3" roughness={0.8} />
        </mesh>
      </Float>
    </group>
  );
}

export default function RobotAssistant() {
  return (
    <div className="w-32 h-32 md:w-40 md:h-40 relative">
      <Canvas
        camera={{ position: [0, 0.5, 3.5], fov: 40 }}
        style={{ background: 'transparent' }}
        gl={{ alpha: true, antialias: true }}
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[2, 3, 2]} intensity={1} castShadow />
        <directionalLight position={[-1, 1, 3]} intensity={0.5} />
        <pointLight position={[0, 2, 2]} intensity={0.8} color="#aaccff" />
        <Robot />
      </Canvas>
    </div>
  );
}