[
    UserPromptNode(
        user_prompt='There is a land that has four cars in it.',
        system_prompts=(
            "You will be given an image description and you will need to call the appropriate tools to get the image recognition results. Use 'count_objects' to get the object count and 'detect_object' to get the detected objects.",
        ),
        system_prompt_functions=[],
        system_prompt_dynamic_functions={}
    ),
    ModelRequestNode(
        request=ModelRequest(
            parts=[
                SystemPromptPart(
                    content="You will be given an image description and you will need to call the appropriate tools to get the image recognition results. Use 'count_objects' to get the object count and 'detect_object' to get the detected objects.",
                    timestamp=datetime.datetime(2025, 4, 17, 9, 30, 9, 688544, tzinfo=datetime.timezone.utc),
                    dynamic_ref=None,
                    part_kind='system-prompt'
                ),
                UserPromptPart(
                    content='There is a land that has four cars in it.',
                    timestamp=datetime.datetime(2025, 4, 17, 9, 30, 9, 688548, tzinfo=datetime.timezone.utc),
                    part_kind='user-prompt'
                )
            ],
            kind='request'
        )
    ),
    CallToolsNode(
        model_response=ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='count_objects',
                    args='{}',
                    tool_call_id='call_nHIWwqwp7dKOfxIKaB29YTWr',
                    part_kind='tool-call'
                ),
                ToolCallPart(
                    tool_name='detect_object',
                    args='{}',
                    tool_call_id='call_Ql8zifm10e6OIMKReGtromPR',
                    part_kind='tool-call'
                )
            ],
            model_name='gpt-4o-2024-08-06',
            timestamp=datetime.datetime(2025, 4, 17, 9, 30, 8, tzinfo=datetime.timezone.utc),
            kind='response'
        )
    ),
    ModelRequestNode(
        request=ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='count_objects',
                    content='Detect the number of objects based on the image description.',
                    tool_call_id='call_nHIWwqwp7dKOfxIKaB29YTWr',
                    timestamp=datetime.datetime(2025, 4, 17, 9, 30, 11, 418139, tzinfo=datetime.timezone.utc),
                    part_kind='tool-return'
                ),
                ToolReturnPart(
                    tool_name='detect_object',
                    content='Detect the object based on the image description.',
                    tool_call_id='call_Ql8zifm10e6OIMKReGtromPR',
                    timestamp=datetime.datetime(2025, 4, 17, 9, 30, 11, 418171, tzinfo=datetime.timezone.utc),
                    part_kind='tool-return'
                )
            ],
            kind='request'
        )
    ),
    CallToolsNode(
        model_response=ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='final_result',
                    args='{"object_count":4,"detected_objects":["cars"]}',
                    tool_call_id='call_zMF6H7x2vRv1R5MKupE2OGqC',
                    part_kind='tool-call'
                )
            ],
            model_name='gpt-4o-2024-08-06',
            timestamp=datetime.datetime(2025, 4, 17, 9, 30, 10, tzinfo=datetime.timezone.utc),
            kind='response'
        )
    ),
    End(
        data=FinalResult(
            data=ImageResult(
                object_count=4,
                detected_objects=['cars']
            ),
            tool_name='final_result',
            tool_call_id='call_zMF6H7x2vRv1R5MKupE2OGqC'
        )
    )
]
