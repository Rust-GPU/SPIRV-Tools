//! Parsing egglog extraction results.

/// Parse the extract result from egglog.
pub fn parse_extract_result(s: &str) -> Option<String> {
    let s = s.trim();

    if !s.starts_with("ExtractBest(") {
        return Some(s.to_string());
    }

    if let Some(nodes_start) = s.find("nodes: {") {
        let rest = &s[nodes_start + 8..];
        if let Some(nodes_end) = rest.find('}') {
            let nodes_str = &rest[..nodes_end];
            let nodes = parse_term_dag_nodes(nodes_str);
            if !nodes.is_empty() {
                return Some(nodes.last()?.clone());
            }
        }
    }

    None
}

fn parse_term_dag_nodes(nodes_str: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut depth = 0;
    let mut current = String::new();

    for c in nodes_str.chars() {
        match c {
            '(' | '[' | '{' => {
                depth += 1;
                current.push(c);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 => {
                let node = current.trim().to_string();
                if !node.is_empty() {
                    if let Some(term) = term_dag_node_to_term(&node, &result) {
                        result.push(term);
                    }
                }
                current.clear();
            }
            _ => current.push(c),
        }
    }

    let node = current.trim().to_string();
    if !node.is_empty() {
        if let Some(term) = term_dag_node_to_term(&node, &result) {
            result.push(term);
        }
    }

    result
}

fn term_dag_node_to_term(node: &str, previous_nodes: &[String]) -> Option<String> {
    let node = node.trim();

    if node.starts_with("Lit(Int(") && node.ends_with("))") {
        let val = &node[8..node.len() - 2];
        return Some(format!("(Const {})", val));
    }

    if node.starts_with("Lit(String(") && node.ends_with("))") {
        let sym = &node[11..node.len() - 2];
        return Some(format!("(Sym {})", sym));
    }

    if node.starts_with("App(") && node.ends_with(')') {
        let inner = &node[4..node.len() - 1];

        if let Some(quote_start) = inner.find('"') {
            if let Some(quote_end) = inner[quote_start + 1..].find('"') {
                let op_name = &inner[quote_start + 1..quote_start + 1 + quote_end];

                if let Some(bracket_start) = inner.find('[') {
                    if let Some(bracket_end) = inner.rfind(']') {
                        let indices_str = &inner[bracket_start + 1..bracket_end];
                        let indices: Vec<usize> = indices_str
                            .split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();

                        if indices.is_empty() {
                            return Some(format!("({})", op_name));
                        }

                        let children: Vec<String> = indices
                            .iter()
                            .map(|&i| {
                                if i < previous_nodes.len() {
                                    previous_nodes[i].clone()
                                } else {
                                    format!("?{}", i)
                                }
                            })
                            .collect();

                        return Some(format!("({} {})", op_name, children.join(" ")));
                    }
                }
            }
        }
    }

    None
}
