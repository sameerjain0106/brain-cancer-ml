export const labels = {
  brain_menin: { name: "Meningioma", color: "#346cb3" },
  brain_tumor: { name: "Tumor", color: "#b04c31" },
  brain_glioma: { name: "Glioma", color: "#0e8363" },
};

export const MODEL_IMAGE_SIDE = 512;
export const BGR_CHANNEL_MEANS = Object.freeze([103.939, 116.779, 123.68]);

export function unitVector(values, dimension) {
  if (values.length !== dimension) throw new Error("The feature vector has the wrong size.");
  let squaredLength = 0;
  for (const value of values) squaredLength += value * value;
  if (!Number.isFinite(squaredLength) || squaredLength <= 0) {
    throw new Error("No usable features came from this image. Try another crop.");
  }
  const scale = 1 / Math.sqrt(squaredLength);
  return Float32Array.from(values, (value) => value * scale);
}

export function rankGroups(query, vectors, count, excludedIndex = -1) {
  if (vectors.length !== query.length * count) throw new Error("The reference index is incomplete.");
  const rows = [];
  for (let index = 0; index < count; index += 1) {
    if (index === excludedIndex) continue;
    const start = index * query.length;
    let dot = 0;
    for (let dimension = 0; dimension < query.length; dimension += 1) {
      dot += query[dimension] * vectors[start + dimension];
    }
    const distance = 1 - Math.max(-1, Math.min(1, dot));
    rows.push({ index, distance });
  }
  rows.sort((first, second) => first.distance - second.distance || first.index - second.index);
  return rows;
}

export function project(query, projection) {
  return projection.components.map((component) => {
    let value = 0;
    for (let index = 0; index < query.length; index += 1) {
      value += (query[index] - projection.mean[index]) * component[index];
    }
    return value;
  });
}

function pngSize(view) {
  if (view.byteLength < 24 || view.getUint32(0) !== 0x89504e47 ||
      view.getUint32(4) !== 0x0d0a1a0a || view.getUint32(8) !== 13 ||
      view.getUint32(12) !== 0x49484452) return null;
  return { type: "image/png", width: view.getUint32(16), height: view.getUint32(20) };
}

function jpegSize(view) {
  if (view.byteLength < 10 || view.getUint16(0) !== 0xffd8) return null;
  let offset = 2;
  while (offset + 3 < view.byteLength) {
    if (view.getUint8(offset) !== 0xff) return null;
    while (offset < view.byteLength && view.getUint8(offset) === 0xff) offset += 1;
    if (offset + 2 >= view.byteLength) return null;
    const marker = view.getUint8(offset);
    offset += 1;
    if (marker === 0xda || marker === 0xd9) return null;
    if (marker === 0x01 || (0xd0 <= marker && marker <= 0xd7)) continue;
    const length = view.getUint16(offset);
    if (length < 2 || offset + length > view.byteLength) return null;
    const isFrame = 0xc0 <= marker && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker);
    if (isFrame && length >= 8) {
      return { type: "image/jpeg", width: view.getUint16(offset + 5), height: view.getUint16(offset + 3) };
    }
    offset += length;
  }
  return null;
}

export async function prepareImage(file) {
  if (file.size > 10 * 1024 * 1024) throw new Error("Choose a JPG or PNG smaller than 10 MB.");
  const view = new DataView(await file.arrayBuffer());
  const format = pngSize(view) ?? jpegSize(view);
  if (!format || !format.width || !format.height) throw new Error("Choose a readable JPG or PNG image.");
  if (format.width * format.height > 4_200_000) {
    throw new Error("Use a crop under 4.2 megapixels. The original project used 512 × 512 MRI crops.");
  }
  const blob = file.slice(0, file.size, format.type);
  const url = URL.createObjectURL(blob);
  let bitmap;
  try {
    let image;
    try {
      bitmap = await createImageBitmap(blob, { imageOrientation: "none", colorSpaceConversion: "none" });
      image = bitmap;
    } catch {
      image = new Image();
      image.src = url;
      await image.decode();
    }
    const width = image.width;
    const height = image.height;
    if (!width || !height || width * height > 4_200_000) throw new Error("The decoded crop is too large.");
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Image decoding is unavailable in this browser.");
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, width, height).data;
    canvas.width = 1;
    canvas.height = 1;
    const side = MODEL_IMAGE_SIDE;
    const plane = side * side;
    const tensor = new Float32Array(3 * plane);
    for (let y = 0; y < side; y += 1) {
      const sourceRow = Math.floor((y + 0.5) * height / side) * width;
      for (let x = 0; x < side; x += 1) {
        const source = (sourceRow + Math.floor((x + 0.5) * width / side)) * 4;
        const target = y * side + x;
        tensor[target] = pixels[source + 2] - BGR_CHANNEL_MEANS[0];
        tensor[plane + target] = pixels[source + 1] - BGR_CHANNEL_MEANS[1];
        tensor[2 * plane + target] = pixels[source] - BGR_CHANNEL_MEANS[2];
      }
    }
    return { url, tensor };
  } catch (error) {
    URL.revokeObjectURL(url);
    throw error;
  } finally {
    bitmap?.close();
  }
}
