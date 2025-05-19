import { sidebar } from "vuepress-theme-hope";

export default sidebar({
  "/": [
    "",
    {
      text: "深度学习",
      icon: "laptop-code",
      prefix: "dl/",
      children: "structure",
    },
    {
      text: "大语言模型",
      icon: "laptop-code",
      prefix: "demo/",
      link: "demo/",
      children: "structure",
    },
    {
      text: "多模态",
      icon: "book",
      prefix: "posts/",
      children: "structure",
    },
    {
      text: "3D-VL",
      icon: "book",
      prefix: "3d-vl/",
      children: "structure",
    },
    {
      text: "开源项目",
      icon: "book",
      prefix: "projects/",
      children: "structure",
    },
  ],
});
