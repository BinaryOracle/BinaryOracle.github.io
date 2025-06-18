import { defineUserConfig } from "vuepress";
import { slimsearchPlugin } from '@vuepress/plugin-slimsearch';
import { commentPlugin } from '@vuepress/plugin-comment';

import theme from "./theme.js";

export default defineUserConfig({
  base: "/",

  lang: "zh-CN",
  title: "MetaMind",
  description: "技术共建，知识共享",

  theme,

  plugins: [
    slimsearchPlugin({
      indexContent: true
    }),
    // commentPlugin({
    //   comment: true,
    //   provider: "Giscus",
    //   repo: "BinaryOracle/BlogComment",
    //   repoId: "R_kgDOOtFoWQ",
    //   category: "General",
    //   categoryId: "DIC_kwDOOtFoWc4CqXdZ",
    // }),
  ],

  // 和 PWA 一起启用
  // shouldPrefetch: false,
});