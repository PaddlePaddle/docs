import argparse
import ast
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any


class Config:
    """配置管理类"""
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.script_dir = Path(__file__).parent
        
        # 文件路径配置
        self.md_file_path = self.base_dir / "pytorch_api_mapping_cn.md"
        self.json_file_path = self.script_dir / "api_difference_info.json"
        self.no_need_convert_path = self.script_dir / "global_var.py"
        self.api_mapping_path = self.script_dir / "api_mapping.json"
        self.api_alias_mapping_path = self.script_dir / "api_alias_mapping.json"
        self.attribute_mapping_path = self.script_dir / "attribute_mapping.json"
        
        # API目录配置
        self.api_dirs = [
            self.base_dir / "api_difference",
            self.base_dir / "api_difference_third_party",
        ]
        
        # 类别定义
        self.all_categories = [
            "API 完全一致", "仅 API 调用方式不一致", "仅参数名不一致",
            "paddle 参数更多", "参数默认值不一致", "torch 参数更多",
            "输入参数用法不一致", "输入参数类型不一致", "返回参数类型不一致",
            "组合替代实现", "可删除", "API 别名", "功能缺失"
        ]
        
        # 特殊类别（前两类）
        self.special_categories = self.all_categories[:2]


class APIMappingProcessor:
    """API映射处理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.existing_apis: Set[str] = set()
        self.docs_mapping: Dict[str, Dict] = {}
        
    def load_mapping_data(self) -> bool:
        """加载映射JSON数据"""
        try:
            with open(self.config.json_file_path, "r", encoding="utf-8") as f:
                mapping_data = json.load(f)
            self.docs_mapping = {item["src_api"]: item for item in mapping_data}
            return True
        except Exception as e:
            print(f"错误: 读取JSON文件 {self.config.json_file_path} 时出错: {e}")
            return False
    
    def escape_underscores(self, api_name: str) -> str:
        """处理API名称中的下划线转义"""
        underscore_count = api_name.count("_")
        return api_name.replace("_", r"\_") if underscore_count >= 2 else api_name
    
    def get_mapping_doc_url(self, torch_api: str) -> str:
        """获取差异对比文档URL"""
        mapping_url_head = "https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/"
        expected_filename = f"{torch_api}.md"
        final_name = f"{torch_api}.html"
        
        for api_dir in self.config.api_dirs:
            for file_path in api_dir.rglob(expected_filename):
                relative_path = file_path.relative_to(self.config.base_dir).with_suffix('.html')
                return f"[差异对比]({mapping_url_head}{relative_path.as_posix()})"
        
        return "-"
    
    def create_api_link(self, api_name: str, url: str) -> str:
        """创建API超链接"""
        if not url:
            return self.escape_underscores(api_name)
        return f"[{self.escape_underscores(api_name)}]({url})"


class FileParser:
    """文件解析器"""
    
    @staticmethod
    def parse_md_files(directories: List[Path]) -> Dict[str, List[Dict]]:
        """解析MD文件获取类别和API信息"""
        category_api_map = defaultdict(list)
        
        for directory in directories:
            for md_file in directory.rglob("*.md"):
                try:
                    with open(md_file, "r", encoding="utf-8") as f:
                        first_line = f.readline().strip()
                    
                    match = re.match(r"##\s*\d*\.?\s*\[(.*?)\](.*)", first_line)
                    if match:
                        category = match.group(1).strip()
                        api_name = match.group(2).strip().replace(r"\_", "_")
                        
                        # 只处理非特殊类别
                        if category not in ["API 完全一致", "仅 API 调用方式不一致"]:
                            category_api_map[category].append({
                                "api_name": api_name,
                                "file_path": md_file
                            })
                    else:
                        print(f"警告: 无法解析文件 {md_file} 的第一行: {first_line}")
                except Exception as e:
                    print(f"错误: 读取文件 {md_file} 时出错: {e}")
        
        return category_api_map
    
    @staticmethod
    def extract_no_need_convert_list(file_path: Path) -> List[str]:
        """提取无需转换的API列表"""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if (isinstance(node, ast.ClassDef) and node.name == "GlobalManager"):
                    for class_node in node.body:
                        if (isinstance(class_node, ast.Assign) and 
                            any(target.id == "NO_NEED_CONVERT_LIST" 
                                for target in class_node.targets if hasattr(target, 'id'))):
                            list_source = ast.get_source_segment(content, class_node.value)
                            return ast.literal_eval(list_source)
        except Exception as e:
            print(f"错误: 解析文件 {file_path} 时出错: {e}")
        
        return []
    
    @staticmethod
    def load_json_file(file_path: Path) -> Dict:
        """加载JSON文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"错误: 读取JSON文件 {file_path} 时出错: {e}")
            return {}


class TableGenerator:
    """表格生成器"""
    
    def __init__(self, processor: APIMappingProcessor, file_parser: FileParser):
        self.processor = processor
        self.file_parser = file_parser
        self.config = processor.config
    
    def generate_table_header(self) -> str:
        """生成表格头部"""
        return "| 序号 | Pytorch 最新 release | Paddle develop | 备注 |\n" \
               "|------|-------------------|---------------|------|"
    
    def generate_table_row(self, idx: int, torch_api: str, paddle_api: str, 
                          torch_url: str = "", paddle_url: str = "", remark: str = "-") -> str:
        """生成表格行"""
        torch_display = self.processor.create_api_link(torch_api, torch_url)
        paddle_display = self.processor.create_api_link(paddle_api, paddle_url)
        return f"| {idx} | {torch_display} | {paddle_display} | {remark} |"
    
    def generate_category1_table(self) -> str:
        """生成类别1（API完全一致）表格"""
        white_list = [
            "torch.Tensor.imag", "torch.Tensor.is_coalesced", "torch.Tensor.is_sparse",
            "torch.Tensor.is_sparse_csr", "torch.Tensor.logical_not_", "torch.Tensor.real",
            "torch.iinfo", "torch.nn.utils.clip_grad_norm_", "torch.nn.utils.clip_grad_value_",
        ]
        
        no_need_list = self.file_parser.extract_no_need_convert_list(self.config.no_need_convert_path)
        rows = []
        used_apis = set()
        
        # 处理无需转换列表中的API
        for torch_api in no_need_list:
            if torch_api in used_apis:
                continue
            
            paddle_api = torch_api.replace("torch", "paddle")
            used_apis.add(torch_api)
            self.processor.existing_apis.add(torch_api)
            
            mapping_info = self.processor.docs_mapping.get(torch_api, {})
            src_url = mapping_info.get("src_api_url")
            
            # 查找paddle_api对应的URL
            dst_url = None
            for item in self.processor.docs_mapping.values():
                if item.get("dst_api") == paddle_api:
                    dst_url = item.get("dst_api_url")
                    break
            
            rows.append((torch_api, paddle_api, src_url, dst_url, "-"))
        
        # 处理映射数据中的额外API
        for src_api, item in self.processor.docs_mapping.items():
            mapping_type = item.get("mapping_type", "")
            dst_api = item.get("dst_api", "")
            
            # if (mapping_type in ["无参数", "参数完全一致"] and  
            #     src_api not in used_apis and 
            #     src_api not in white_list):
                
            #     expected_paddle_api = src_api.replace("torch", "paddle")
            #     if expected_paddle_api == dst_api:
            #         used_apis.add(src_api)
            #         self.processor.existing_apis.add(src_api)
                    
            #         src_url = item.get("src_api_url")
            #         dst_url = item.get("dst_api_url")
            #         rows.append((src_api, dst_api, src_url, dst_url, "-"))
        
        return self._build_table_from_rows(rows)
    
    def generate_category2_table(self) -> str:
        """生成类别2（仅API调用方式不一致）表格"""
        whitelist_skip = [
            "torch.Tensor.numel", "torch.Tensor.nelement", "torch.Tensor.is_inference",
            "torch.numel", "torch.is_inference", "torch.ge",
            "torch.utils.data.WeightedRandomSampler", "torch.utils.data.RandomSampler",
        ]
        
        no_need_list = self.file_parser.extract_no_need_convert_list(self.config.no_need_convert_path)
        api_mapping_data = self.file_parser.load_json_file(self.config.api_mapping_path)
        attribute_mapping_data = self.file_parser.load_json_file(self.config.attribute_mapping_path)
        api_mapping_data.update(attribute_mapping_data)
        
        rows = []
        used_apis = set()
        
        # 处理api_mapping中的特定Matcher类型
        for src_api, mapping_info in api_mapping_data.items():
            if src_api in whitelist_skip or src_api in no_need_list:
                continue
            
            matcher = mapping_info.get("Matcher", "")
            if matcher in ["ChangeAPIMatcher", "TensorFunc2PaddleFunc", "Func2Attribute", "Attribute2Func"]:
                docs_info = self.processor.docs_mapping.get(src_api, {})
                src_url = docs_info.get("src_api_url")
                paddle_api = mapping_info.get("paddle_api") or docs_info.get("dst_api", "")
                
                # 查找paddle_api对应的URL
                dst_url = None
                for item in self.processor.docs_mapping.values():
                    if item.get("dst_api") == paddle_api:
                        dst_url = item.get("dst_api_url")
                        break
                
                remark = self.processor.get_mapping_doc_url(src_api)
                rows.append((src_api, paddle_api, src_url, dst_url, remark))
                used_apis.add(src_api)
                self.processor.existing_apis.add(src_api)
        
        # 处理映射数据中的不一致API
        for src_api, item in self.processor.docs_mapping.items():
            if (src_api in whitelist_skip or src_api in no_need_list or 
                src_api in used_apis):
                continue
            
            mapping_type = item.get("mapping_type", "")
            dst_api = item.get("dst_api", "")
            
            if (mapping_type in ["仅 API 调用方式不一致"]):
                expected_paddle_api = rc_api.replace("torch", "paddle")
                if expected_paddle_api != dst_api:
                    used_apis.add(src_api)
                    self.processor.existing_apis.add(src_api)
                    
                    src_url = item.get("src_api_url")
                    dst_url = item.get("dst_api_url")
                    remark = self.processor.get_mapping_doc_url(src_api)
                    rows.append((src_api, dst_api, src_url, dst_url, remark))
        
        return self._build_table_from_rows(rows)
    
    def generate_category12_table(self) -> str:
        """生成类别12（API别名映射）表格"""
        api_alias_data = self.file_parser.load_json_file(self.config.api_alias_mapping_path)
        rows = []
        
        for torch_api, torch_api_alias in api_alias_data.items():
            if torch_api in self.processor.existing_apis:
                continue
            
            mapping_info = self.processor.docs_mapping.get(torch_api_alias, {})
            dst_api = mapping_info.get("dst_api", "-")
            dst_url = mapping_info.get("dst_api_url", "")
            src_url = self.processor.docs_mapping.get(torch_api, {}).get("src_api_url", "")
            
            torch_api_alias_display = torch_api_alias.replace(r'\_', '_')
            url = self.processor.get_mapping_doc_url(torch_api_alias)
            remark = f"``{torch_api_alias_display}`` 别名， {url}"
            rows.append((torch_api, dst_api, src_url, dst_url, remark))
            
            self.processor.existing_apis.update([torch_api, torch_api_alias])
        
        return self._build_table_from_rows(rows)
    
    def generate_category13_table(self, md_content: str) -> str:
        """生成类别13（功能缺失）表格"""
        pattern = r"### 13\. 功能缺失([\s\S]*?)(?=### |$)"
        match = re.search(pattern, md_content)
        if not match:
            return self._build_empty_table()
        
        section_content = match.group(1)
        table_pattern = r"\| 序号 \| Pytorch 最新 release \| Paddle develop \| 备注 \|\n\|[-\| ]+\|\n([\s\S]*?)(?=\n\n|\Z)"
        table_match = re.search(table_pattern, section_content)
        
        if not table_match:
            return self._build_empty_table()
        
        table_content = table_match.group(1)
        rows = []
        
        for line in table_content.split("\n"):
            if not line.startswith("|"):
                continue
            
            parts = line.split("|")
            if len(parts) < 5:
                continue
            
            torch_api_cell = parts[2].strip()
            paddle_api_cell = parts[3].strip()
            remark_cell = parts[4].strip()
            
            # 提取Torch API名称
            torch_api_match = re.match(r"\[(.*?)\]\(.*?\)", torch_api_cell)
            torch_api = torch_api_match.group(1) if torch_api_match else torch_api_cell
            
            if torch_api in self.processor.existing_apis:
                continue
            
            # 提取原Torch URL
            torch_link_match = re.search(r"\((.*?)\)", torch_api_cell)
            torch_url = torch_link_match.group(1) if torch_link_match else ""
            
            # 查找Paddle映射
            mapping_info = self.processor.docs_mapping.get(torch_api, {})
            paddle_api = mapping_info.get("dst_api", "-")
            paddle_url = mapping_info.get("dst_api_url", "")
            
            rows.append((torch_api, paddle_api, torch_url, paddle_url, remark_cell))
            self.processor.existing_apis.add(torch_api)
        
        return self._build_table_from_rows(rows)
    
    def generate_regular_category_table(self, category: str, api_list: List[Dict]) -> str:
        """生成常规类别（3-11）表格"""
        rows = []
        
        for api_info in api_list:
            api_name = api_info["api_name"]
            
            if api_name in self.processor.existing_apis:
                continue
            
            mapping_info = self.processor.docs_mapping.get(api_name, {})
            dst_api = mapping_info.get("dst_api", "-")
            
            if dst_api == "暂无" or not dst_api:
                dst_api = "-"
            
            src_url = mapping_info.get("src_api_url", "")
            dst_url = mapping_info.get("dst_api_url", "")
            remark = self.processor.get_mapping_doc_url(api_name)
            
            rows.append((api_name, dst_api, src_url, dst_url, remark))
            self.processor.existing_apis.add(api_name)
        
        return self._build_table_from_rows(rows) if rows else self._build_empty_table()
    
    def _build_table_from_rows(self, rows: List[Tuple]) -> str:
        """从行数据构建完整表格"""
        table_lines = [self.generate_table_header()]
        
        for idx, (torch_api, paddle_api, torch_url, paddle_url, remark) in enumerate(rows, 1):
            table_lines.append(self.generate_table_row(idx, torch_api, paddle_api, torch_url, paddle_url, remark))
        
        return "\n".join(table_lines)
    
    def _build_empty_table(self) -> str:
        """构建空表格"""
        return f"{self.generate_table_header()}\n| 1 | - | - | 新增中...... |"


class DocumentUpdater:
    """文档更新器"""
    
    def __init__(self, config: Config, table_generator: TableGenerator):
        self.config = config
        self.table_generator = table_generator
    
    def add_category_numbers(self, md_content: str) -> str:
        """为类别标题添加序号"""
        updated_content = md_content
        for idx, category in enumerate(self.config.all_categories, 1):
            pattern = rf"### \d*\.?\s*{re.escape(category)}"
            replacement = f"### {idx}. {category}"
            updated_content = re.sub(pattern, replacement, updated_content)
        return updated_content
    
    def update_special_category_table(self, md_content: str, category: str, table_content: str) -> str:
        """更新特殊类别表格"""
        pattern = rf"(### \d*\.?\s*{re.escape(category)}[\s\S]*?)(\| 序号 \| Pytorch 最新 release \| Paddle develop \| 备注 \|\n\|[-\| ]+\|\n)[\s\S]*?(?=### \d*\.?\s*|\Z)"
        replacement = rf"\1{table_content}\n\n"
        return re.sub(pattern, replacement, md_content, flags=re.MULTILINE)
    
    def update_regular_category_table(self, md_content: str, category: str, table_content: str) -> str:
        """更新常规类别表格"""
        pattern = rf"(### \d*\.?\s*{re.escape(category)}[\s\S]*?)(\| 序号 \| Pytorch 最新 release \| Paddle develop \| 备注 \|\n\|[-\| ]+\|\n)[\s\S]*?(?=### \d*\.?\s*|\Z)"
        replacement = rf"\1{table_content}\n\n"
        return re.sub(pattern, replacement, md_content, flags=re.MULTILINE)
    
    def update_document(self, check_mode: bool = False) -> bool:
        """更新主文档"""
        try:
            # 读取原始文档
            with open(self.config.md_file_path, "r", encoding="utf-8") as f:
                original_content = f.read()
            
            # 添加类别序号
            updated_content = self.add_category_numbers(original_content)
            
            # 生成并更新特殊类别表格
            category1_table = self.table_generator.generate_category1_table()
            category2_table = self.table_generator.generate_category2_table()
            
            updated_content = self.update_special_category_table(
                updated_content, "API 完全一致", category1_table)
            updated_content = self.update_special_category_table(
                updated_content, "仅 API 调用方式不一致", category2_table)
            
            print(f"信息: 从前两个特殊类别中总共解析出 {len(self.table_generator.processor.existing_apis)} 个API用于重复检查")
            
            # 解析MD文件获取类别和API信息
            category_api_map = FileParser.parse_md_files(self.config.api_dirs)
            
            # 更新常规类别（3-11）
            for idx, category in enumerate(self.config.all_categories, 1):
                if 3 <= idx <= 11 and category in category_api_map:
                    table_content = self.table_generator.generate_regular_category_table(
                        category, category_api_map[category])
                    updated_content = self.update_regular_category_table(
                        updated_content, category, table_content)
            
            # 更新类别12和13
            category12_table = self.table_generator.generate_category12_table()
            category13_table = self.table_generator.generate_category13_table(updated_content)
            
            updated_content = self.update_special_category_table(
                updated_content, "API 别名", category12_table)
            updated_content = self.update_special_category_table(
                updated_content, "功能缺失", category13_table)
            
            # 确定输出文件
            output_file = self.config.script_dir / "tmp_check.md" if check_mode else self.config.md_file_path
            
            # 写入更新后的内容
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(updated_content)
            
            print(f"成功: 文档已更新到 {output_file}")
            return True
            
        except Exception as e:
            print(f"错误: 更新文档时出错: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="更新PyTorch到PaddlePaddle API映射文档")
    parser.add_argument("--check", action="store_true", help="检查模式，输出到临时文件")
    args = parser.parse_args()
    
    # 获取基础目录
    base_dir = Path(__file__).parent.parent
    config = Config(base_dir)
    
    # 初始化处理器
    processor = APIMappingProcessor(config)
    if not processor.load_mapping_data():
        return
    
    # 初始化文件解析器和表格生成器
    file_parser = FileParser()
    table_generator = TableGenerator(processor, file_parser)
    
    # 更新文档
    updater = DocumentUpdater(config, table_generator)
    success = updater.update_document(args.check)
    
    if not success:
        print("错误: 文档更新失败")


if __name__ == "__main__":
    main()