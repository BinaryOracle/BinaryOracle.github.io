import { defineUserConfig } from "vuepress";

import theme from "./theme.js";

export default defineUserConfig({
  base: "/",

  lang: "zh-CN",
  title: "MetaMind",
  description: "探索AI边界,拥抱智能未来",

  theme,

  // 和 PWA 一起启用
  // shouldPrefetch: false,
});
