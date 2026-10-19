"""Evaluate OpenMask3D masks/features with official OpenScan material labels."""
import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Working copy OpenScan root')
    p.add_argument('--gt', type=Path, required=True, help='gt_material directory')
    p.add_argument('--masks', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--clip-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New evidence directory')
    args = p.parse_args()
    for name in ['source', 'gt', 'masks', 'features', 'clip_checkpoint', 'output']:
        setattr(args, name, getattr(args, name).resolve())
    directory = args.source / 'Evaluation/OpenMask3D/openmask3d/evaluation'
    scenes_file = directory / 'val_scenes_scannet200.txt'
    scenes = scenes_file.read_text().splitlines()
    if len(scenes) != 312 or len(set(scenes)) != 312:
        p.error('Expected the official complete 312-scene validation list')
    files = [args.clip_checkpoint, scenes_file]
    for scene in scenes:
        files.extend([args.gt/(scene+'.txt'), args.masks/(scene+'_masks.pt'), args.features/(scene+'_openmask3d_features.npy')])
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        p.error(f'Missing {len(missing)} inputs; partial evaluation refused: {missing[:8]}')
    if args.output.exists():
        p.error('Choose a new output directory; history will not be overwritten')
    if args.output == args.source or args.source in args.output.parents:
        p.error('Keep evidence outside the source working copy')
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(directory))
    # Explicitly select the official material variants; no source file edits.
    import openscan_constants_material as constants
    import eval_semantic_instance_material as evaluator
    sys.modules['scannet_constants'] = constants
    sys.modules['eval_semantic_instance'] = evaluator
    import run_eval_close_vocab_inst_seg as pipeline
    import clip
    pipeline.InstSegEvaluator.get_clip_model = lambda self, kind: clip.load(str(args.clip_checkpoint), self.device)[0]
    pipeline.InstSegEvaluator.get_query_sentences = lambda self, dataset_type, sentence_structure: ['this term is made of '+label for label in constants.CLASS_LABELS_200]
    # Stream hashes instead of reading large masks/checkpoints into RAM.
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                h.update(chunk)
        return h.hexdigest()
    files.extend(directory.glob('*.py'))
    files.append(Path(__file__).resolve())
    (args.output/'inputs.json').write_text(json.dumps([{'path':str(f),'sha256':digest(f)} for f in files], indent=2), encoding='utf-8')
    original = evaluator.evaluate
    observed = {}
    def record(preds, *rest, **kwargs):
        if set(preds) != set(scenes):
            raise RuntimeError('Incomplete scene predictions')
        result = original(preds, *rest, **kwargs)
        metrics = result[0]
        observed.update(ap=100*float(metrics['all_ap']), ap50=100*float(metrics['all_ap_50%']), ap25=100*float(metrics['all_ap_25%']), scenes=312, unit='percent', aspect='material', query_template='this term is made of {}')
        return result
    pipeline.evaluate = record
    with (args.output/'evaluation.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        pipeline.test_pipeline_full_scannet200(str(args.features), str(args.gt), str(args.masks), '{}', '{}_openmask3d_features.npy', scene_list_file=str(scenes_file))
    (args.output/'metrics.json').write_text(json.dumps(observed, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(observed, indent=2))
    print('Metrics are run observations; paper agreement requires protocol/provenance review.')


if __name__ == '__main__':
    main()
