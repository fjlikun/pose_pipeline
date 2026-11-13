"""Validate per-video notebook settings before creating output directories."""
import os
from pathlib import Path
from multi_person import normalize_manual_box_specs, validate_person_limit


def prepare_video_jobs(inputs,*,max_people=20,manual_bboxes=None):
    default_limit=validate_person_limit(max_people)
    items=[inputs] if isinstance(inputs,(str,os.PathLike,dict)) else list(inputs)
    if not items: raise ValueError('VIDEOS 至少需要一个视频路径')
    jobs=[]
    for item in items:
        config=item if isinstance(item,dict) else {'video':item}
        unknown=set(config)-{'video','max_people','bboxes'}
        if unknown or 'video' not in config:
            raise ValueError('每个视频可设置 video、max_people、bboxes，且必须提供 video')
        video=Path(config['video']).expanduser().resolve()
        if not video.is_file(): raise FileNotFoundError(video)
        limit=validate_person_limit(config.get('max_people',default_limit))
        boxes=config.get('bboxes',manual_bboxes)
        if boxes is not None:
            import av
            with av.open(str(video)) as container:
                if not container.streams.video: raise ValueError(f'No video stream: {video}')
                stream=container.streams.video[0]
                boxes=normalize_manual_box_specs(boxes,stream.codec_context.width,stream.codec_context.height)
        jobs.append(dict(video=video,max_people=limit,bboxes=boxes))
    return jobs
