import { hopeTheme } from "vuepress-theme-hope";

import navbar from "./navbar.js";
//import sidebar from "./sidebar.js";

export default hopeTheme({
  hostname: "https://mister-hope.github.io",

  author: {
    name: "MetaMind",
    url: "https://blog.csdn.net/m0_53157173",
    email: "zdhdhyzdhdhy@gmail.com",
  },
  
  logo: "assets/images/head.png",

  docsDir: "src",

  // 导航栏
  navbar,

  // 侧边栏
  sidebar: "structure",

  // 页脚
  footer: "默认页脚",
  displayFooter: true,

  // 博客相关
  blog: {
    description: "探索AI边界,拥抱智能未来",
    intro: "/intro.html",
    medias: {
      csdn: "https://blog.csdn.net/m0_53157173?type=blog",
      GitHub: "https://github.com/BinaryOracle",
      Gmail: "mailto:info@example.com",
    },
  },

  // 加密配置
  encrypt: {
    config: {
      "/demo/encrypt.html": {
        hint: "Password: 1234",
        password: "1234",
      },
    },
  },

  // 多语言配置
  metaLocales: {
    editLink: "在 GitHub 上编辑此页",
  },

  // 如果想要实时查看任何改变，启用它。注: 这对更新性能有很大负面影响
  // hotReload: true,

  // 此处开启了很多功能用于演示，你应仅保留用到的功能。
  markdown: {
    align: true,
    attrs: true,
    codeTabs: true,
    component: true,
    demo: true,
    figure: true,
    gfm: true,
    imgLazyload: true,
    imgMark: true,
    imgSize: true,
    include: true,
    mark: true,
    plantuml: true,
    spoiler: true,
    stylize: [
      {
        matcher: "Recommended",
        replacer: ({ tag }) => {
          if (tag === "em")
            return {
              tag: "Badge",
              attrs: { type: "tip" },
              content: "Recommended",
            };
        },
      },
    ],
    sub: true,
    sup: true,
    tabs: true,
    tasklist: true,
    vPre: true,
    math: {
      type: "mathjax",
    },
  },

  pageInfo: ["Author", "Original", "Date", "Category", "Tag", "ReadingTime","Word"],

  // fullscreen: true,
  // focus: true,
  
  // 在这里配置主题提供的插件
  plugins: {
    blog: true,
    
    components: {
      components: ["Badge", "VPCard"],
    },

    icon: {
      prefix: "fa6-solid:",
    },
  },
});
