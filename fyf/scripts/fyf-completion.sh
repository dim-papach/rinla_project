#!/usr/bin/env bash
# fyf-completion.bash - Bash completion for FYF

_fyf_completion() {
    local cur prev opts commands
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # Main commands
    commands="simulate process pipeline validate plot"
    
    # Global options
    global_opts="--verbose --quiet --version --help"
    
    # Options for each command
    simulate_opts="--cosmic-fraction --cosmic-value --cosmic-seed --trails --trail-width --min-angle --max-angle --trail-value --output-dir --report"
    process_opts="--shape --mesh-cutoff --tolerance --restart --scaling --nonstationary --output-dir --check-inla --install-inla --generate-plots --report"
    pipeline_opts="--cosmic-fraction --cosmic-value --cosmic-seed --trails --trail-width --min-angle --max-angle --trail-value --shape --mesh-cutoff --tolerance --restart --scaling --nonstationary --output-dir --skip-plots --dpi --cmap --report"
    validate_opts="--output-dir --plot --html-report --dpi --cmap"
    plot_opts="--plot-type --output-dir --dpi --cmap --residual-cmap --percentile-min --percentile-max --residual-percentile-min --residual-percentile-max"
    
    # If the previous word is a flag expecting a value, provide appropriate completions
    case "$prev" in
        --shape)
            COMPREPLY=( $(compgen -W "none radius ellipse" -- "$cur") )
            return 0
            ;;
        --plot-type)
            COMPREPLY=( $(compgen -W "comparison residual all" -- "$cur") )
            return 0
            ;;
        --cmap|--residual-cmap)
            COMPREPLY=( $(compgen -W "viridis plasma inferno magma cividis twilight coolwarm bwr" -- "$cur") )
            return 0
            ;;
        --output-dir|-o)
            # Complete directory names
            COMPREPLY=( $(compgen -d -- "$cur") )
            return 0
            ;;
        # Add more options here that need specific completions
    esac
    
    # Figure out which command we're completing for
    local command=""
    for ((i=1; i < COMP_CWORD; i++)); do
        if [[ "${COMP_WORDS[i]}" == @(simulate|process|pipeline|validate|plot) ]]; then
            command="${COMP_WORDS[i]}"
            break
        fi
    done
    
    # If no command has been specified yet, complete commands and global options
    if [[ -z "$command" ]]; then
        if [[ "$cur" == -* ]]; then
            COMPREPLY=( $(compgen -W "$global_opts" -- "$cur") )
        else
            COMPREPLY=( $(compgen -W "$commands" -- "$cur") )
        fi
        return 0
    fi
    
    # Complete options for the specific command
    case "$command" in
        simulate)
            if [[ "$cur" == -* ]]; then
                COMPREPLY=( $(compgen -W "$simulate_opts" -- "$cur") )
            else
                # Complete FITS files for arguments
                COMPREPLY=( $(compgen -f -X '!*.[fF][iI][tT][sS]' -- "$cur") )
            fi
            ;;
        process)
            if [[ "$cur" == -* ]]; then
                COMPREPLY=( $(compgen -W "$process_opts" -- "$cur") )
            else
                # Complete FITS files for arguments
                COMPREPLY=( $(compgen -f -X '!*.[fF][iI][tT][sS]' -- "$cur") )
            fi
            ;;
        pipeline)
            if [[ "$cur" == -* ]]; then
                COMPREPLY=( $(compgen -W "$pipeline_opts" -- "$cur") )
            else
                # Complete FITS files for arguments
                COMPREPLY=( $(compgen -f -X '!*.[fF][iI][tT][sS]' -- "$cur") )
            fi
            ;;
        validate)
            if [[ "$cur" == -* ]]; then
                COMPREPLY=( $(compgen -W "$validate_opts" -- "$cur") )
            else
                # Complete FITS files for arguments
                COMPREPLY=( $(compgen -f -X '!*.[fF][iI][tT][sS]' -- "$cur") )
            fi
            ;;
        plot)
            if [[ "$cur" == -* ]]; then
                COMPREPLY=( $(compgen -W "$plot_opts" -- "$cur") )
            else
                # Complete FITS files for arguments
                COMPREPLY=( $(compgen -f -X '!*.[fF][iI][tT][sS]' -- "$cur") )
            fi
            ;;
    esac
    
    return 0
}

# Register the completion function
complete -F _fyf_completion fyf