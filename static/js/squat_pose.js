/**
 * MediaPipe Pose → POST /api/predict con las columnas de modelosupervisado2.py
 * (6 ángulos: rodillas, caderas, tronco, media de rodilla). Orden = window.__FEATURE_NAMES__.
 */
import {
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const IDX = {
  L_SH: 11,
  R_SH: 12,
  L_HIP: 23,
  R_HIP: 24,
  L_KNEE: 25,
  R_KNEE: 26,
  L_ANK: 27,
  R_ANK: 28,
  /** dedos del pie (BlazePose) para ángulo en tobillo rodilla-tobillo-dedo */
  L_FOOT_IDX: 31,
  R_FOOT_IDX: 32,
};

function angleAtDeg(a, b, c) {
  const bax = a.x - b.x,
    bay = a.y - b.y;
  const bcx = c.x - b.x,
    bcy = c.y - b.y;
  const nba = Math.hypot(bax, bay);
  const nbc = Math.hypot(bcx, bcy);
  if (nba < 1e-6 || nbc < 1e-6) return null;
  const cos = Math.max(-1, Math.min(1, (bax * bcx + bay * bcy) / (nba * nbc)));
  return (Math.acos(cos) * 180) / Math.PI;
}

function trunkInclinationDeg(lm) {
  const ls = lm[IDX.L_SH],
    rs = lm[IDX.R_SH];
  const lh = lm[IDX.L_HIP],
    rh = lm[IDX.R_HIP];
  const msx = (ls.x + rs.x) * 0.5,
    msy = (ls.y + rs.y) * 0.5;
  const mhx = (lh.x + rh.x) * 0.5,
    mhy = (lh.y + rh.y) * 0.5;
  const vx = msx - mhx,
    vy = msy - mhy;
  const n = Math.hypot(vx, vy);
  if (n < 1e-6) return null;
  const cos = Math.max(-1, Math.min(1, (vx * 0 + vy * -1) / n));
  return (Math.acos(cos) * 180) / Math.PI;
}

function meanVis(lm, indices) {
  let s = 0;
  for (const i of indices) {
    const v = lm[i].visibility ?? 0;
    s += v;
  }
  return s / indices.length;
}

/**
 * Ángulo en tobillo (rodilla–tobillo–punta del pie) y distancias normalizadas
 * entre rodillas / entre tobillos (0–1 en el plano de la imagen).
 * Solo para debug o un futuro reentrenamiento; el modelo actual no las consume.
 */
function computeNotebookStyleExtras(lm, visMin = 0.35) {
  const need = [
    IDX.L_KNEE,
    IDX.R_KNEE,
    IDX.L_ANK,
    IDX.R_ANK,
    IDX.L_FOOT_IDX,
    IDX.R_FOOT_IDX,
  ];
  for (const i of need) {
    if ((lm[i]?.visibility ?? 0) < visMin) return null;
  }
  const la = angleAtDeg(lm[IDX.L_KNEE], lm[IDX.L_ANK], lm[IDX.L_FOOT_IDX]);
  const ra = angleAtDeg(lm[IDX.R_KNEE], lm[IDX.R_ANK], lm[IDX.R_FOOT_IDX]);
  if (la == null || ra == null) return null;
  const lk = lm[IDX.L_KNEE],
    rk = lm[IDX.R_KNEE];
  const laa = lm[IDX.L_ANK],
    raa = lm[IDX.R_ANK];
  const dist_knees = Math.hypot(lk.x - rk.x, lk.y - rk.y);
  const dist_ankles = Math.hypot(laa.x - raa.x, laa.y - raa.y);
  return {
    left_ankle_angle: la,
    right_ankle_angle: ra,
    dist_knees,
    dist_ankles,
  };
}

function extractFeatures(lm, names, visMin = 0.35) {
  const need = [
    IDX.L_SH,
    IDX.R_SH,
    IDX.L_HIP,
    IDX.R_HIP,
    IDX.L_KNEE,
    IDX.R_KNEE,
    IDX.L_ANK,
    IDX.R_ANK,
  ];
  for (const i of need) {
    if ((lm[i].visibility ?? 0) < visMin) return null;
  }

  const lk = angleAtDeg(lm[IDX.L_HIP], lm[IDX.L_KNEE], lm[IDX.L_ANK]);
  const rk = angleAtDeg(lm[IDX.R_HIP], lm[IDX.R_KNEE], lm[IDX.R_ANK]);
  const lh = angleAtDeg(lm[IDX.L_SH], lm[IDX.L_HIP], lm[IDX.L_KNEE]);
  const rh = angleAtDeg(lm[IDX.R_SH], lm[IDX.R_HIP], lm[IDX.R_KNEE]);
  const trunk = trunkInclinationDeg(lm);
  if (lk == null || rk == null || lh == null || rh == null || trunk == null)
    return null;

  const kneeMean = (lk + rk) * 0.5;
  const hipMean = (lh + rh) * 0.5;
  const symmetry = (Math.abs(lk - rk) + Math.abs(lh - rh)) * 0.5;
  const visibility_score = meanVis(lm, need);

  const map = {
    left_knee_angle: lk,
    right_knee_angle: rk,
    left_hip_angle: lh,
    right_hip_angle: rh,
    trunk_inclination: trunk,
    knee_angle_mean: kneeMean,
  };

  for (const n of names) {
    if (map[n] === undefined) return null;
  }
  const features = names.map((n) => map[n]);

  const ankle = computeNotebookStyleExtras(lm, visMin);
  const extras = {
    ...(ankle || {}),
    hip_angle_mean: hipMean,
    visibility_score,
    left_right_symmetry: symmetry,
  };

  return { features, extras };
}

const badge = document.getElementById("badge");
const statusEl = document.getElementById("status");
const video = document.getElementById("cam");
const canvas = document.getElementById("out");
const ctx = canvas.getContext("2d");

const names = window.__FEATURE_NAMES__;
if (!names || !Array.isArray(names) || names.length === 0) {
  badge.textContent = "Error: feature_names";
  badge.className = "badge bad";
}

let poseLandmarker = null;
let lastPredict = 0;
let inflight = false;

/**
 * Debug consola: (A) vector enviado al modelo, (B) métricas extra solo informativas.
 */
function logPayloadParaModelo(features, extras) {
  if (!names || !Array.isArray(names) || names.length === 0) {
    console.warn("[sentadillas] sin __FEATURE_NAMES__ válido");
    return;
  }
  const porNombre = Object.fromEntries(
    names.map((n, i) => [n, Number(features[i])])
  );
  console.groupCollapsed("[sentadillas] → POST /api/predict (modelo entrenado)");
  console.log(`features (array ${features.length} valores):`, features);
  console.log("por_nombre (objeto):", porNombre);
  console.log(
    "[sentadillas] nombre : valor\n" +
      names.map((n, i) => `  ${n}: ${features[i]}`).join("\n")
  );
  console.log("JSON enviado al modelo:", JSON.stringify({ features }));
  console.table(porNombre);
  console.groupEnd();

  if (extras && Object.keys(extras).length > 0) {
    console.groupCollapsed(
      "[sentadillas] extras (no enviados al modelo; solo depuración)"
    );
    console.table(extras);
    console.groupEnd();
  }
}

async function predictApi(features, extras) {
  logPayloadParaModelo(features, extras);
  const r = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });
  if (!r.ok) throw new Error(await r.text());
  const data = await r.json();
  console.log("[sentadillas] ← respuesta modelo:", data);
  return data;
}

function setBadge(etiqueta, ok) {
  badge.textContent = etiqueta;
  badge.classList.remove("neutral", "ok", "bad");
  if (ok === true) badge.classList.add("ok");
  else if (ok === false) badge.classList.add("bad");
  else badge.classList.add("neutral");
}

async function main() {
  statusEl.textContent = "Cargando MediaPipe…";
  const wasm = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(wasm, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
      delegate: "CPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  const resize = () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  };
  video.addEventListener("loadeddata", resize);
  resize();

  statusEl.textContent = "Listo. Colócate de perfil o tres cuartos.";

  function loop() {
    if (video.readyState >= 2) {
      const ts = performance.now();
      const res = poseLandmarker.detectForVideo(video, ts);
      ctx.save();
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.restore();

      let etiqueta = "Sin detección";
      let ok = null;

      if (res.landmarks && res.landmarks.length > 0) {
        const lm = res.landmarks[0];
        const pack = extractFeatures(lm, names);
        if (pack && !inflight && ts - lastPredict > 120) {
          inflight = true;
          lastPredict = ts;
          predictApi(pack.features, pack.extras)
            .then((d) => {
              setBadge(d.etiqueta, d.ok);
            })
            .catch((e) => {
              statusEl.textContent = String(e);
            })
            .finally(() => {
              inflight = false;
            });
        }
        if (!pack) {
          setBadge("Puntos poco visibles", null);
        }
      } else {
        setBadge("Sin detección", null);
      }
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

main().catch((e) => {
  setBadge("Error", false);
  statusEl.textContent = String(e);
});
