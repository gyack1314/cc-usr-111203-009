"""显示已有原始摘要，或运行相关灾害连通风险分析；来源文件保持不动。"""
import argparse
import json

from .domain import summarize
from .io import load
from .risk.analysis import analyze


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    inspect_p = sub.add_parser('inspect', help='输出原始资料摘要')
    inspect_p.add_argument('file')
    risk_p = sub.add_parser('risk', help='相关灾害下的终端连通风险分析')
    risk_p.add_argument('file')
    risk_p.add_argument('-o', '--output', help='报告输出路径，缺省打印到标准输出')
    a = parser.parse_args()
    v, digest = load(a.file)
    if a.command == 'inspect':
        payload = {'source_sha256': digest, 'summary': summarize(v)}
    else:
        payload = analyze(v, digest)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    if a.command == 'risk' and a.output:
        with open(a.output, 'w', encoding='utf8') as fh:
            fh.write(text)
    else:
        print(text)


if __name__ == '__main__':
    main()
