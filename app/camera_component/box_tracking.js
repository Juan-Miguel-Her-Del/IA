/* Follow the detector's box between replies by matching its image texture. */
const BOX_MAX_AGE = 1100;
const UNTRACKED_MAX_AGE = 650;

function boxOverlap(a, b) {
  const area = box => (box[2] - box[0]) * (box[3] - box[1]);
  const intersection = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) *
    Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  return intersection / (area(a) + area(b) - intersection);
}

function boxTemplate(frame, box) {
  const {width, height, pixels} = frame;
  const sampleX = [], sampleY = [], values = [];
  for (let row = 0; row < 12; row++) {
    for (let col = 0; col < 12; col++) {
      const x = Math.round((box[0] + (box[2] - box[0]) * (.12 + .76 * col / 11)) * (width - 1));
      const y = Math.round((box[1] + (box[3] - box[1]) * (.12 + .76 * row / 11)) * (height - 1));
      sampleX.push(x);
      sampleY.push(y);
      values.push(pixels[y * width + x]);
    }
  }
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const centered = Float32Array.from(values, value => value - mean);
  const energy = centered.reduce((sum, value) => sum + value * value, 0);
  if (energy < 64 * values.length || sampleX[11] - sampleX[0] < 5 ||
      sampleY.at(-1) - sampleY[0] < 5) return null;
  return {sampleX, sampleY, centered, energy, mean, box, width, height};
}

class CameraBoxTrack {
  constructor(detection, frame, observedAt) {
    this.detection = detection;
    this.box = detection.box;
    this.template = boxTemplate(frame, this.box);
    this.observedAt = observedAt;
    this.startedAt = observedAt;
    this.dx = 0;
    this.dy = 0;
    this.misses = 0;
    this.unmatchedFrames = 0;
  }

  isFresh(now) {
    const limit = this.template ? BOX_MAX_AGE : UNTRACKED_MAX_AGE;
    return now - this.observedAt < limit;
  }

  follow(frame) {
    if (!this.template) return false;
    const {sampleX, sampleY, centered, energy, mean, box, width, height} = this.template;
    if (frame.width !== width || frame.height !== height) return false;
    const pixels = frame.pixels;
    let bestScore = -Infinity, bestX = this.dx, bestY = this.dy;
    const consider = (dx, dy) => {
      if (sampleX[0] + dx < 0 || sampleX.at(-1) + dx >= width ||
          sampleY[0] + dy < 0 || sampleY.at(-1) + dy >= height) return;
      let sum = 0, squares = 0, product = 0, residual = 0;
      for (let i = 0; i < centered.length; i++) {
        const pixel = pixels[(sampleY[i] + dy) * width + sampleX[i] + dx];
        sum += pixel;
        squares += pixel * pixel;
        product += centered[i] * pixel;
        residual += Math.abs(pixel - (centered[i] + mean));
      }
      const variance = squares - sum * sum / centered.length;
      if (variance < energy * .35 || residual / centered.length > 30) return;
      const correlation = product / Math.sqrt(energy * variance);
      const score = correlation - .0015 * (Math.abs(dx - this.dx) + Math.abs(dy - this.dy));
      if (score > bestScore) {
        bestScore = score;
        bestX = dx;
        bestY = dy;
      }
    };
    for (let dy = this.dy - 8; dy <= this.dy + 8; dy += 2) {
      for (let dx = this.dx - 8; dx <= this.dx + 8; dx += 2) consider(dx, dy);
    }
    const coarseX = bestX, coarseY = bestY;
    for (let dy = coarseY - 1; dy <= coarseY + 1; dy++) {
      for (let dx = coarseX - 1; dx <= coarseX + 1; dx++) consider(dx, dy);
    }
    if (bestScore < .76) return false;
    this.dx = bestX;
    this.dy = bestY;
    this.box = box.map((value, i) => {
      const offset = i % 2 === 0 ? bestX / (width - 1) : bestY / (height - 1);
      return Math.max(0, Math.min(1, value + offset));
    });
    return true;
  }
}
