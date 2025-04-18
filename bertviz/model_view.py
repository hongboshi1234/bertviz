import json
import os
import uuid

from IPython.display import display, HTML, Javascript

from .util import format_special_chars, format_attention, format_attention_all, num_layers, num_heads
# from bertviz.util import pad_2d_tensor, add_padding, format_attention_all, num_layers, num_heads

def model_view(
        attention=None,
        tokens=None,
        sentence_b_start=None,
        prettify_tokens=True,
        display_mode="dark",
        encoder_attention=None,
        decoder_attention=None,
        cross_attention=None,
        encoder_tokens=None,
        decoder_tokens=None,
        include_layers=None,
        include_heads=None,
        html_action='view'
):
    """Render model view

        Args:
            For self-attention models:
                attention: list of ``torch.FloatTensor``(one for each layer) of shape
                    ``(batch_size(must be 1), num_heads, sequence_length, sequence_length)``
                tokens: list of tokens
                sentence_b_start: index of first wordpiece in sentence B if input text is sentence pair (optional)
            For encoder-decoder models:
                encoder_attention: list of ``torch.FloatTensor``(one for each layer) of shape
                    ``(batch_size(must be 1), num_heads, encoder_sequence_length, encoder_sequence_length)``
                decoder_attention: list of ``torch.FloatTensor``(one for each layer) of shape
                    ``(batch_size(must be 1), num_heads, decoder_sequence_length, decoder_sequence_length)``
                cross_attention: list of ``torch.FloatTensor``(one for each layer) of shape
                    ``(batch_size(must be 1), num_heads, decoder_sequence_length, encoder_sequence_length)``
                encoder_tokens: list of tokens for encoder input
                decoder_tokens: list of tokens for decoder input
            For all models:
                prettify_tokens: indicates whether to remove special characters in wordpieces, e.g. Ġ
                display_mode: 'light' or 'dark' display mode
                include_layers: indices (zero-based) of layers to include in visualization. Defaults to all layers.
                    Note: filtering layers may improve responsiveness of the visualization for long inputs.
                include_heads: indices (zero-based) of heads to include in visualization. Defaults to all heads.
                    Note: filtering heads may improve responsiveness of the visualization for long inputs.
                html_action: Specifies the action to be performed with the generated HTML object
                    - 'view' (default): Displays the generated HTML representation as a notebook cell output
                    - 'return' : Returns an HTML object containing the generated view for further processing or custom visualization
    """

    attn_data = []
    n_heads=num_heads(attention)
    n_layers=num_layers(attention)
    if attention is not None:
        if tokens is None:
            raise ValueError("'tokens' is required")
        if encoder_attention is not None or decoder_attention is not None or cross_attention is not None \
                or encoder_tokens is not None or decoder_tokens is not None:
            raise ValueError("If you specify 'attention' you may not specify any encoder-decoder arguments. This"
                             " argument is only for self-attention models.")
        # n_heads = num_heads(attention)
        if include_layers is None:
            include_layers = list(range(n_layers))
        if include_heads is None:
            include_heads = list(range(n_heads))
        attention = format_attention_all(attention, include_layers, include_heads)
        if sentence_b_start is None:
            # attention = format_attention_all(attention, include_layers, include_heads)
            attn_data.append(
                {
                    'name': None,
                    'attn': attention.tolist(),
                    'left_text': tokens,
                    'right_text': tokens
                }
            )
        else:
            slice_a = slice(0, sentence_b_start)  # Positions corresponding to sentence A in input
            slice_b = slice(sentence_b_start, len(tokens))  # Position corresponding to sentence B in input
            attn_data.append(
                {
                    'name': 'Sentence B -> Sentence A',
                    'attn': attention[:, :, slice_b, slice_a].tolist(),
                    'left_text': tokens[slice_b],
                    'right_text': tokens[slice_a]
                }
            )

    elif encoder_attention is not None or decoder_attention is not None or cross_attention is not None:
        if encoder_attention is not None:
            if encoder_tokens is None:
                raise ValueError("'encoder_tokens' required if 'encoder_attention' is not None")
            if include_layers is None:
                include_layers = list(range(n_layers))
            n_heads = num_heads(encoder_attention)
            if include_heads is None:
                include_heads = list(range(n_heads))
            encoder_attention = format_attention(encoder_attention, include_layers, include_heads)
            attn_data.append(
                {
                    'name': 'Encoder',
                    'attn': encoder_attention.tolist(),
                    'left_text': encoder_tokens,
                    'right_text': encoder_tokens
                }
            )
        if decoder_attention is not None:
            if decoder_tokens is None:
                raise ValueError("'decoder_tokens' required if 'decoder_attention' is not None")
            if include_layers is None:
                include_layers = list(range(n_layers))
            n_heads = num_heads(decoder_attention)
            if include_heads is None:
                include_heads = list(range(n_heads))
            decoder_attention = format_attention(decoder_attention, include_layers, include_heads)
            attn_data.append(
                {
                    'name': 'Decoder',
                    'attn': decoder_attention.tolist(),
                    'left_text': decoder_tokens,
                    'right_text': decoder_tokens
                }
            )
        if cross_attention is not None:
            if encoder_tokens is None:
                raise ValueError("'encoder_tokens' required if 'cross_attention' is not None")
            if decoder_tokens is None:
                raise ValueError("'decoder_tokens' required if 'cross_attention' is not None")
            if include_layers is None:
                include_layers = list(range(n_layers))
            n_heads = num_heads(cross_attention)
            if include_heads is None:
                include_heads = list(range(n_heads))
            cross_attention = format_attention_all(cross_attention, include_layers, include_heads)
            attn_data.append(
                {
                    'name': 'Cross',
                    'attn': cross_attention.tolist(),
                    'left_text': decoder_tokens,
                    'right_text': encoder_tokens
                }
            )
    else:
        raise ValueError("You must specify at least one attention argument.")

    # Generate unique div id to enable multiple visualizations in one notebook
    vis_id = 'bertviz-%s'%(uuid.uuid4().hex)

    # Compose html
    if len(attn_data) > 1:
        options = '\n'.join(
            f'<option value="{i}">{attn_data[i]["name"]}</option>'
            for i, d in enumerate(attn_data)
        )
        select_html = f'Attention: <select id="filter">{options}</select>'
    else:
        select_html = ""
    vis_html = f"""      
        <div id="{vis_id}" style="font-family:'Helvetica Neue', Helvetica, Arial, sans-serif;">
            <span style="user-select:none">
                {select_html}
            </span>
            <div id='vis'></div>
        </div>
    """

    for d in attn_data:
        attn_seq_len_left = len(d['attn'][0][0])
        if attn_seq_len_left != len(d['left_text']):
            raise ValueError(
                f"Attention has {attn_seq_len_left} positions, while number of tokens is {len(d['left_text'])} "
                f"for tokens: {' '.join(d['left_text'])}"
            )
        attn_seq_len_right = len(d['attn'][0][0][0])
        if attn_seq_len_right != len(d['right_text']):
            raise ValueError(
                f"Attention has {attn_seq_len_right} positions, while number of tokens is {len(d['right_text'])} "
                f"for tokens: {' '.join(d['right_text'])}"
            )
        if prettify_tokens:
            d['left_text'] = format_special_chars(d['left_text'])
            d['right_text'] = format_special_chars(d['right_text'])
    print(f'attn_data.shape:{len(d["attn"])}, {len(d["attn"][0])}, {len(d["attn"][0][0])}, {len(d["attn"][0][0][0])}') 
    print(f'n_heads:{n_heads}')
    print(f'include_layers:{include_layers}')
    params = {
        'attention': attn_data,
        'default_filter': "0",
        'display_mode': display_mode,
        'root_div_id': vis_id,
        'include_layers': include_layers,
        'include_heads': include_heads,
        'total_heads': n_heads
    }

    # require.js must be imported for Colab or JupyterLab:
    if html_action == 'view':
        display(HTML('<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>'))
        display(HTML(vis_html))
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))
        vis_js = open(os.path.join(__location__, 'model_view.js')).read().replace("PYTHON_PARAMS", json.dumps(params))
        display(Javascript(vis_js))

    elif html_action == 'return':
        html1 = HTML('<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>')

        html2 = HTML(vis_html)

        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))
        vis_js = open(os.path.join(__location__, 'model_view.js')).read()
        # Remove the RequireJS config from the JS file as we'll handle it in the HTML
        vis_js = vis_js[vis_js.find('requirejs(['): ]
        vis_js = vis_js.replace("PYTHON_PARAMS", json.dumps(params))
        html3 = Javascript(vis_js)
        script = '\n<script type="text/javascript">\n' + html3.data + '\n</script>\n'

        # Create a complete HTML document with all required scripts
        complete_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Attention Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/5.7.0/d3.min.js"></script>
    <style>
        .dark {{
            background-color: #222;
            color: #fff;
        }}
        .light {{
            background-color: #fff;
            color: #222;
        }}
        .attention-head {{
            position: relative;
            margin-bottom: 5px;
        }}
        .attention-head-text {{
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            font-size: 12px;
        }}
        .attention-head-map {{
            margin-left: 100px;
        }}
        #vis {{
            padding: 20px;
        }}
    </style>
</head>
<body class="{display_mode}">
    {vis_html}
    <script>
    require.config({{
        paths: {{
            d3: 'https://cdnjs.cloudflare.com/ajax/libs/d3/5.7.0/d3.min',
            jquery: 'https://cdnjs.cloudflare.com/ajax/libs/jquery/3.3.1/jquery.min'
        }}
    }});
    
    const params = {json.dumps(params)};
    </script>
    <script type="text/javascript">
    {vis_js}
    </script>
    <script>
    // Initialize visualization after page load
    window.addEventListener('load', function() {{
        require(['d3', 'jquery'], function(d3, $) {{
            window.d3 = d3;
            window.jQuery = $;
            window.$ = $;
            
            // Initialize config object
            const config = {{}};
            config.attention = params.attention;
            config.filter = params.default_filter;
            config.rootDivId = params.root_div_id;
            config.layers = params.include_layers;
            config.heads = params.include_heads;
            config.totalHeads = params.total_heads;
            
            // Call render function
            render();
        }});
    }});
    </script>
</body>
</html>
"""
        # Save as standalone HTML file
        with open('attention_visualization.html', 'w', encoding='utf-8') as f:
            f.write(complete_html)
        
        # Also return the IPython HTML object for notebook display
        head_html = HTML(html1.data + html2.data + script)
        return head_html

    else:
        raise ValueError("'html_action' parameter must be 'view' or 'return")