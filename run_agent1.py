from agents.ingestion_agent1 import run_ingestion
from core.config import agent1_config as config

if __name__ == '__main__':
    state = {"input_dir": config.DATASETS_DIR}
    out = run_ingestion(state)
    print('\n== Agent1 output summary ==')
    print('Processed:', len(out.get('raw_docs', [])))
    print('Extracted files (sample 10):')
    for p in out.get('extracted_files', [])[:10]:
        print(' -', p)
    print('\nDocuments info path:', out.get('documents_info_json_path'))
    print('Trace CSV path:', out.get('trace_csv_path'))
