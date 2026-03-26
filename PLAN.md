# Show Case Website Plan

## Goal

在 `/data/yfyang/show_case_web` 目录下做一个正式的展示网站，用于呈现：

- `RealRestorer` 品牌和定位
- `ranking.txt` 里的排名结果
- 所有参与排名模型的 logo / wordmark 顶部展示
- 多个 restoration 任务的 before / after 可视化
- 首页视频占位区
- 一个比 `examples.html` 更真实、更完整、直接使用你现有素材的网站

这次先输出计划，不直接开始正式页面开发，等你确认后再落地。

## Existing Materials Audit

### Existing reference

- 示例页：`/data/yfyang/show_case_web/examples.html`
- Logo：`/data/yfyang/show_case_web/asset/RealRestorer_logo.png`
- Ranking 数据：`/data/yfyang/show_case_web/ranking.txt`

### Current showcase categories

- `blur`：14 张
- `compression`：8 张
- `flare`：12 张
- `haze`：12 张
- `lowlight`：14 张
- `moire`：12 张
- `noise`：12 张
- `old_phtoto`：10 张
- `rain`：16 张
- `reflection`：16 张
- `snow`：10 张
- `water`：2 张

### Observed data issues

- 存在命名不规则文件，例如 `blur/blur_0020_output .png` 文件名里有空格。
- `old_phtoto` 目录名有拼写问题，但不影响使用；正式网站里建议展示名写成 `Old Photo Restoration`。
- 不同目录里的输入输出命名规则并不完全统一，需要做一层数据映射，而不是靠纯字符串规则盲猜。
- 当前没有看到可直接用于首页 hero 的 demo 视频素材；`examples.html` 里的视频是占位方案。
- 当前目录里没有其他参赛模型的 logo 资源，后续需要补充一层 logo 资产或统一生成 wordmark 卡片兜底。

## Product Direction

## Proposed site type

建议做成一个单页静态展示站，主文件结构控制在：

- `index.html`
- `styles.css`
- `app.js`
- `data.js` 或 `showcase-data.json`

原因：

- 你的素材已经基本够做完整 landing page
- 单页最适合科研/模型展示
- 后续部署最简单，直接静态托管即可

## Proposed information architecture

### 1. Hero

内容：

- `RealRestorer` logo
- 一句核心定位
- 简短副标题
- 一个视频展示占位区域
- 入口按钮

建议按钮：

- `View Ranking`
- `Browse Results`
- `Method Overview`

视频部分先不接真实媒体文件，保留一个样式完整的 placeholder 容器，后面你替换 mp4 或嵌入 demo 即可。

### 2. Ranking section

这是你最开始强调要有的部分，建议放在首页靠前位置。

展示方式：

- 不做普通表格
- 采用柱状图排名主视觉
- 只使用 `ranking.txt` 最后一列数值
- 顶部先展示所有模型 logo / wordmark
- 柱状图里突出 `RealRestorer (ours)`，但不伪造排序

建议做法：

- 从 `ranking.txt` 读取每个模型的最后一列数值
- 默认按最后一列从高到低排序，做成横向柱状图
- 每个柱条左侧放模型名，顶部独立做一排 logo 卡片
- 对 `RealRestorer (ours)` 使用品牌色和特殊描边
- 柱状图区域只保留一个 metric label，避免信息过载

logo 方案：

- 优先使用各模型的官方 logo / 品牌标识
- 如果某些模型没有可用官方 logo，就用统一风格的 typographic wordmark 兜底
- 所有 logo 最终会被整理到站点本地 `asset/` 目录，避免线上依赖

当前默认假设：

- 最后一列就是正式展示指标
- 数值越大，排名越高

还需要你补一个信息：

- 最后一列指标的正式名字

### 3. Method / value proposition section

内容：

- RealRestorer 的一段简洁介绍
- 支持的 restoration task 标签云
- 1 到 3 个模型亮点

建议只写产品级表述，不先写过深的论文叙述。

### 4. Task showcase section

主体部分建议按任务分类展示。

每个任务模块包含：

- 任务标题
- 一句简短说明
- 1 个大尺寸交互 before/after 对比器
- 下方若干缩略图，点击切换示例

建议首批展示任务：

- Blur Removal
- Low-light Enhancement
- Dehazing
- Reflection Removal
- Rain Removal
- Denoising
- Compression Artifact Removal
- Moire Removal
- Lens Flare Removal
- Old Photo Restoration
- Snow Removal
- Water Degradation Removal

### 5. Footer

内容：

- 品牌名
- 联系方式或代码链接占位
- 页面版权信息

## Visual Direction

不建议照搬 `examples.html` 那套偏“通用 AI 官网”的渐变风格。

建议方向：

- 整体气质更偏 research showcase / imaging system
- 颜色以暖白、深灰、少量高饱和电蓝或青绿色作为强调
- 让图片成为主角，不要让背景和装饰抢走注意力
- 排名模块要有“结果可信”的感觉，而不是营销页

具体风格建议：

- Hero 背景用浅色渐层 + 微弱网格或扫描线纹理
- 排名卡片用偏硬朗的数据可视化语言
- Showcase 区域用大留白 + 清晰的图片容器
- Hover 动效克制，重点放在 before/after 对比交互

## Data Handling Plan

建议不要在第一版里直接自动扫描所有图片并现场猜配对。

更稳妥的方案：

1. 我先在站点里建立一份手工整理的数据表。
2. 每个任务写明确的样例对。
3. 页面从这份数据表渲染。

好处：

- 避免命名混乱导致错配
- 你后续增删样例更可控
- 能处理 `jpg/png/webp` 混合格式

建议数据结构示意：

```js
{
  taskId: "blur",
  title: "Blur Removal",
  description: "Recover sharp edges and local textures from real blurry inputs.",
  samples: [
    {
      label: "Sample 01",
      input: "./blur/blur_0001_input.png",
      output: "./blur/blur_0001_output.png"
    }
  ]
}
```

## Implementation Plan

### Phase 1. Data curation

- 整理 ranking 数据，只保留最后一列
- 为所有模型补齐 logo 资产或 wordmark 兜底
- 为每个任务挑 4 到 8 组代表性样例
- 手工写入展示数据文件
- 规避坏命名和异常文件

### Phase 2. Page skeleton

- 建立 `index.html`
- 搭出 hero / logo wall / ranking / showcase / footer 结构
- 放入真实 logo 和真实数据
- 预留视频 placeholder 区域

### Phase 3. Interaction

- 实现 before/after slider
- 实现缩略图切换
- 实现任务导航或 sticky category tabs
- 为柱状图增加轻量入场动画

### Phase 4. Polish

- 统一图片比例与容器表现
- 移动端适配
- 加载体验和过渡动画
- 检查文件路径和浏览器兼容性
- 检查 logo 视觉统一性和版权标识方式

## Files To Be Created After Confirmation

- `/data/yfyang/show_case_web/index.html`
- `/data/yfyang/show_case_web/styles.css`
- `/data/yfyang/show_case_web/app.js`
- `/data/yfyang/show_case_web/showcase-data.js`
- `/data/yfyang/show_case_web/ranking-data.js`

可选：

- `/data/yfyang/show_case_web/asset/site-preview.png`
- `/data/yfyang/show_case_web/asset/model-logos/*`

## Decisions Needed From You

### Must confirm

1. `ranking.txt` 最后一列指标的正式名字是什么。

### If you do not specify, I will assume

- 单页静态站
- 首页保留视频 placeholder
- 排名区按最后一列降序生成柱状图
- 顶部展示所有模型 logo；无官方 logo 时用统一风格 wordmark
- 每个任务展示 4 到 6 组样例
- 正式站点文件直接输出到 `/data/yfyang/show_case_web`

## Recommended Next Step

如果这个计划没问题，我下一步会直接开始实现第一版正式网站，并输出到：

- `/data/yfyang/show_case_web/index.html`
- `/data/yfyang/show_case_web/styles.css`
- `/data/yfyang/show_case_web/app.js`
- `/data/yfyang/show_case_web/showcase-data.js`

然后你可以直接本地打开检查。
