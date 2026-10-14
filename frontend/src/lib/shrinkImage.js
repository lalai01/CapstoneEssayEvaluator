// Phone photos are often 3–5 MB, slow to upload on mobile data. The server
// reads pages at up to 2400 px on the long side (that's what Gemini gets),
// so shrink bigger photos in the browser first: same accuracy, far less to
// upload. Anything that can't be decoded here (e.g. TIFF) is sent unchanged.
const MAX_SIDE = 2400;
const SMALL_ENOUGH = 1.5 * 1024 * 1024;

export default async function shrinkImage(file) {
  if (!/^image\/(jpeg|png|webp|bmp)$/.test(file.type)) return file;
  let bitmap;
  try {
    // 'from-image' applies the photo's EXIF rotation, so pages stay upright.
    bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch {
    return file;
  }
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  if (scale === 1 && file.size <= SMALL_ENOUGH) {
    bitmap.close?.();
    return file;
  }
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#fff';   // transparent PNG areas become white paper, not black
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close?.();
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.88));
  if (!blob || blob.size >= file.size) return file;
  const name = file.name.replace(/\.[^.]+$/, '') + '.jpg';
  return new File([blob], name, { type: 'image/jpeg' });
}
