from fastapi.testclient import TestClient
from app.server import app

client = TestClient(app)
samples = ['sample_1.jpg', 'sample_2.jpg', 'sample_3.jpg', 'sample_4.jpg', 'sample_5.jpg', 'sample_6.jpg']

for s in samples:
    r = client.post(
        '/api/segment',
        data={
            'sample_name': s,
            'conf_threshold': 0.25,
            'model_version': 'ensemble',
            'classifier_mode': 'ensemble',
            'calibration_preset': 'bruvs_rig',
            'pixels_per_cm': 22.0
        }
    ).json()

    fish_dets = [d for d in r.get('detections', []) if d.get('class_name') == 'fish']
    print(f"\n==================== {s} (Total Fish: {len(fish_dets)}) ====================")
    for i, f in enumerate(fish_dets):
        tax = f.get('taxonomy', {})
        bio = f.get('biomass', {})
        print(f"  [Fish #{i+1}] {tax.get('family')} ({tax.get('common_name')}) - Confidence: {tax.get('confidence')}")
        print(f"    Consensus: {tax.get('agreement_status')}")
        print(f"    Votes: {tax.get('model_votes')}")
        print(f"    Dimensions: Fork Length = {bio.get('fork_length_cm')} cm, Depth = {bio.get('body_depth_cm')} cm")
        print(f"    Biomass: {bio.get('biomass_g')} g ({bio.get('weight_kg')} kg) | Guild: {tax.get('trophic_guild')}")

    if fish_dets:
        cm = r.get('community_metrics', {})
        print(f"  >> Community Biomass: {cm.get('total_biomass_kg')} kg | Commercial Yield: {cm.get('commercial_biomass_kg')} kg ({cm.get('commercial_yield_pct')}%) | Shannon H': {cm.get('shannon_diversity_h')}")
