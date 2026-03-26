window.showcaseData = [
  {
    id: "blur",
    title: "Blur Removal",
    description: "Recover sharp structures, text edges, and local detail from motion blur and focus errors.",
    accent: "#145af2",
    samples: [
      { label: "Street", input: "./blur/blur_0023_input.jpg", output: "./blur/blur_0023_output.jpg" },
      { label: "Motorbike", input: "./blur/blur_45_input.jpg", output: "./blur/blur_45_output.jpg" },
      { label: "Cheetah", input: "./blur/blur_new1_input.jpg", output: "./blur/blur_new1_output.png" },
      { label: "Baseball", input: "./blur/blur_new_2_input.jpg", output: "./blur/new_output.png" }
    ]
  },
  {
    id: "compression",
    title: "Compression Artifact Removal",
    description: "Clean ringing, blocking, and aggressive web-compression artifacts while holding natural contrast.",
    accent: "#1f8bff",
    samples: [
      { label: "Forest Bridge", input: "./compression/input (2).jpg", output: "./compression/0207_sftfiltered_lr1e5_cosine3000_bs4_it1260500 (3).jpg" },
      { label: "Bridge Daylight", input: "./compression/input (4).jpg", output: "./compression/0207_sftfiltered_lr1e5_cosine3000_bs4_it1260500 (5).jpg" },
      { label: "Bridge Night", input: "./compression/input (5).jpg", output: "./compression/0207_sftfiltered_lr1e5_cosine3000_bs4_it1260500 (4).jpg" },
      { label: "Portrait", input: "./compression/new1_input.jpg", output: "./compression/new1_output.webp" }
    ]
  },
  {
    id: "flare",
    title: "Lens Flare Removal",
    description: "Suppress veiling flare and retain image contrast in scenes with strong light sources.",
    accent: "#13b3a8",
    samples: [
      { label: "Flare 01", input: "./flare/flare_0001_input.png", output: "./flare/flare_0001_output.png" },
      { label: "Flare 03", input: "./flare/flare_0003_input.jpg", output: "./flare/flare_0003_output.jpg" },
      { label: "Flare 04", input: "./flare/flare_0004_input.png", output: "./flare/flare_0004_output.png" },
      { label: "New Case", input: "./flare/new_1_input.webp", output: "./flare/new_1_output.webp" }
    ]
  },
  {
    id: "haze",
    title: "Dehazing",
    description: "Lift atmospheric haze while recovering depth cues, neutral colors, and mid-tone separation.",
    accent: "#0c9d7a",
    samples: [
      { label: "Haze 02", input: "./haze/haze_0002_input.png", output: "./haze/haze_0002_output.png" },
      { label: "Haze 04", input: "./haze/haze_0004_input.png", output: "./haze/haze_0004_output.png" },
      { label: "Haze 05", input: "./haze/haze_0005_input.png", output: "./haze/haze_0005_output.png" },
      { label: "Haze 08", input: "./haze/haze_0008_input.png", output: "./haze/haze_0008_output.png" },
    ]
  },
  {
    id: "lowlight",
    title: "Low-light Enhancement",
    description: "Recover visibility in underexposed scenes without flattening the mood or clipping bright regions.",
    accent: "#f0a81d",
    samples: [
      { label: "Flower", input: "./lowlight/lowlight_0003_input.png", output: "./lowlight/lowlight_0003_output.png" },
      { label: "Fruit", input: "./lowlight/lowlight_0005_input.png", output: "./lowlight/lowlight_0005_output.png" },
      { label: "Couple", input: "./lowlight/lowlight_0009_input.jpg", output: "./lowlight/lowlight_0009_output.jpg" },
      { label: "Profile", input: "./lowlight/lowlight_0011_input.png", output: "./lowlight/lowlight_0011_output.png" },
      { label: "Silhouette", input: "./lowlight/new_input.jpg", output: "./lowlight/low_output.webp" }
    ]
  },
  {
    id: "moire",
    title: "Moire Removal",
    description: "Suppress structured interference patterns while preserving fabric, display, and grid detail.",
    accent: "#ab4cff",
    samples: [
      { label: "Pattern 02", input: "./moire/input (2).jpg", output: "./moire/0207_sftfiltered_lr1e5_cosine3000_bs4_it1260500 (2).jpg" },
      { label: "Pattern 03", input: "./moire/input (3).jpg", output: "./moire/0207_sftfiltered_lr1e5_cosine3000_bs4_it1260500 (3).jpg" },
      { label: "Case 01", input: "./moire/moire_new1_input.jpg", output: "./moire/moire_new1_output.jpg" },
      { label: "Portrait", input: "./moire/new_input.jpg", output: "./moire/new_output.jpg" }
    ]
  },
  {
    id: "noise",
    title: "Denoising",
    description: "Remove visible sensor noise while maintaining texture, edges, and color stability.",
    accent: "#3a6df0",
    samples: [
      { label: "Noise 01", input: "./noise/noise_0001_input.png", output: "./noise/noise_0001_output.png" },
      { label: "Noise 03", input: "./noise/noise_0003_input.png", output: "./noise/noise_0003_output.png" },
      { label: "Noise 05", input: "./noise/noise_0005_input.png", output: "./noise/noise_0005_output.png" },
      { label: "Noise New", input: "./noise/noise_new1_input.png", output: "./noise/noise_new1_output.png" },
    ]
  },
  {
    id: "old-photo",
    title: "Old Photo Restoration",
    description: "Repair faded, degraded historical imagery while keeping the recovered result visually plausible.",
    accent: "#8f5e2a",
    samples: [
      { label: "Old 03", input: "./old_phtoto/oldphoto_0003_input.jpg", output: "./old_phtoto/oldphoto_0003_output.jpg" },
      { label: "Old 04", input: "./old_phtoto/oldphoto_0004_input.jpg", output: "./old_phtoto/oldphoto_0004_output.jpg" },
      { label: "Old 10", input: "./old_phtoto/oldphoto_0010_input.jpg", output: "./old_phtoto/oldphoto_0010_output.jpg" },
      { label: "Old 11", input: "./old_phtoto/oldphoto_0011_input.jpg", output: "./old_phtoto/oldphoto_0011_output.jpg" }
    ]
  },
  {
    id: "rain",
    title: "Rain Removal",
    description: "Suppress rain streaks and rainy scene washout while keeping structural detail intact.",
    accent: "#0092c7",
    samples: [
      { label: "Rain 09", input: "./rain/rain_0009_input.jpg", output: "./rain/rain_0009_output.jpg" },
      { label: "Rain 12", input: "./rain/rain_0012_input.png", output: "./rain/rain_0012_output.png" },
      { label: "Rain 13", input: "./rain/rain_0013_input.png", output: "./rain/rain_0013_output.png" },
      { label: "Rain 15", input: "./rain/rain_0015_input.png", output: "./rain/rain_0015_output.png" },
      { label: "Portrait", input: "./rain/new_input.webp", output: "./rain/new_output.webp" }
    ]
  },
  {
    id: "reflection",
    title: "Reflection Removal",
    description: "Separate transmitted content from glass reflections without destroying true scene contrast.",
    accent: "#0f8aa8",
    samples: [
      { label: "Reflection 01", input: "./reflection/reflection_0001_input.png", output: "./reflection/reflection_0001_output.png" },
      { label: "Reflection 07", input: "./reflection/reflection_0007_input.png", output: "./reflection/reflection_0007_output.png" },
      { label: "Reflection 11", input: "./reflection/reflection_0011_input.png", output: "./reflection/reflection_0011_output.png" },
      { label: "Case 02", input: "./reflection/reflection_new2_input.png", output: "./reflection/reflection_new2_output.png" },
    ]
  },
  {
    id: "snow",
    title: "Snow Removal",
    description: "Reduce dense snow occlusion while restoring scene visibility and edge continuity.",
    accent: "#5da9ff",
    samples: [
      { label: "Snow 01", input: "./snow/snow_0001_input.jpg", output: "./snow/snow_0001_output.jpg" },
      { label: "Snow 02", input: "./snow/snow_0002_input.jpg", output: "./snow/snow_0002_output.jpg" },
      { label: "Snow 04", input: "./snow/snow_0004_input.jpg", output: "./snow/snow_0004_output.jpg" },
      { label: "Snow 10", input: "./snow/snow_0010_input.jpg", output: "./snow/snow_0010_output.jpg" },
      { label: "Snow 17", input: "./snow/snow_0017_input.jpg", output: "./snow/snow_0017_output.jpg" }
    ]
  },
 
];
